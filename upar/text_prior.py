"""Attribute head initialized from CLIP text (training only; the result is an ordinary linear head).

Each attribute has a positive and a negative description. Their CLIP text embeddings
(averaged over templates) give one direction per attribute in CLIP's joint space. On
frozen pretrained image features of seen-domain training images, the head is then fitted
in two steps:

1. text logit_i = a_i * <e, d_i> + b_i, with only a scale and a bias per attribute;
2. a residual linear probe on top of the fixed text logits, with a strong L2 penalty.

On LODO (frozen backbone) this beat an equally tuned plain linear probe on all three
held-out domains (mean held-out mADM 0.441 vs 0.416). The two steps fold into one
1024 -> 40 linear layer on the CLS feature (e = feature @ visual.proj), so the model,
checkpoints and submission format are unchanged.
"""
import numpy as np
import torch

from .attributes import ATTRIBUTES

TEMPLATES = ('a photo of {}.', 'a low resolution photo of {}.', 'a cropped photo of {}.',
             'a surveillance camera photo of {}.')
PERSON = 'a person'
COLORS = ('black', 'blue', 'brown', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow')

PROMPTS = {
    'Age-Young': ('a child or teenager', 'an adult'),
    'Age-Adult': ('an adult', 'a child or an elderly person'),
    'Age-Old': ('an elderly person', 'a young person'),
    'Gender-Female': ('a woman', 'a man'),
    # 'short haircut' beat 'short hair' on seen-domain train images (AUC 0.50-0.73 -> 0.71-0.84).
    'Hair-Length-Short': ('a person with a short haircut', 'a person with long hair'),
    'Hair-Length-Long': ('a person with long hair', 'a person with a short haircut'),
    'Hair-Length-Bald': ('a bald person', 'a person with hair'),
    'UpperBody-Length-Short': ('a person wearing a short-sleeved top', 'a person wearing a long-sleeved top'),
    **{f'UpperBody-Color-{c.title()}': (f'a person wearing a {c} top', 'a person wearing a top') for c in COLORS},
    'UpperBody-Color-Other': ('a person wearing a patterned top', 'a person wearing a plain top'),
    'LowerBody-Length-Short': ('a person wearing shorts or a short skirt', 'a person wearing long trousers'),
    **{f'LowerBody-Color-{c.title()}': (f'a person wearing {c} trousers', 'a person wearing trousers') for c in COLORS},
    'LowerBody-Color-Other': ('a person wearing patterned trousers', 'a person wearing plain trousers'),
    'LowerBody-Type-Trousers&Shorts': ('a person wearing trousers or shorts', 'a person wearing a skirt or a dress'),
    'LowerBody-Type-Skirt&Dress': ('a person wearing a skirt or a dress', 'a person wearing trousers or shorts'),
    'Accessory-Backpack': ('a person carrying a backpack', PERSON),
    'Accessory-Bag': ('a person carrying a bag', PERSON),
    'Accessory-Glasses-Normal': ('a person wearing glasses', PERSON),
    'Accessory-Glasses-Sun': ('a person wearing sunglasses', PERSON),
    'Accessory-Hat': ('a person wearing a hat', PERSON),
}
assert tuple(PROMPTS) == ATTRIBUTES


@torch.inference_mode()
def text_directions(state, prompts=PROMPTS, templates=TEMPLATES, source='clip_openai_l14'):
    """Unit-norm (positive - negative) text direction per attribute, shape [40, 768]."""
    from .clip_text import Tokenizer, build_text_encoder
    encoder, tokenizer = build_text_encoder(state, source), Tokenizer()

    def embed(phrases):
        tokens = tokenizer([t.format(p) for p in phrases for t in templates])
        e = torch.nn.functional.normalize(encoder(tokens), dim=-1).view(len(phrases), len(templates), -1)
        return torch.nn.functional.normalize(e.mean(1), dim=-1)

    positive = embed([prompts[a][0] for a in ATTRIBUTES])
    negative = embed([prompts[a][1] for a in ATTRIBUTES])
    return (positive - negative).numpy()


def fit_logistic(x, y, l2=1e-4, offset=None, steps=500):
    """Independent logistic regressions for the columns of y [N, K] on x [N, D], with LBFGS.

    Inputs are standardized so the L2 penalty treats features alike; `offset` [N, K] is a
    fixed logit added before the sigmoid. Returns weight [D, K] and bias [K] for raw inputs.
    """
    mean, std = x.mean(0), x.std(0) + 1e-8
    xs = torch.as_tensor((x - mean) / std, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.float64)
    fixed = torch.zeros_like(target) if offset is None else torch.as_tensor(offset, dtype=torch.float64)
    weight = torch.zeros(xs.shape[1], target.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(target.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn='strong_wolfe', tolerance_change=1e-12)

    def closure():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(fixed + xs @ weight + bias, target)
                + l2 * (weight ** 2).sum() / target.shape[1])
        loss.backward()
        return loss
    optimizer.step(closure)
    weight = weight.detach().numpy() / std[:, None]
    return weight, bias.detach().numpy() - mean @ weight


def calibrate(scores, y, l2=1e-4):
    """Scale and bias per attribute on a 1-D score: logit_i = a_i * score_i + b_i."""
    a, b = np.zeros(scores.shape[1]), np.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        w, c = fit_logistic(scores[:, i:i + 1], y[:, i:i + 1], l2)
        a[i], b[i] = w[0, 0], c[0]
    return a, b


def fit_text_prior(embeddings, labels, directions, l2=0.1):
    """Text-prior head in CLIP's joint space: weight [768, 40] and bias [40] on embeddings [N, 768]."""
    scores = embeddings @ directions.T
    a, b = calibrate(scores, labels)
    text_logits = a * scores + b
    residual, residual_bias = fit_logistic(embeddings, labels, l2, offset=text_logits)
    return directions.T * a + residual, b + residual_bias


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


@torch.inference_mode()
def clip_embeddings(model, proj, paths, preprocess, device, batch_size, workers, amp=False):
    """Projected CLIP image embeddings (CLS feature @ visual.proj) of a frozen model, [N, 768]."""
    from torch.utils.data import DataLoader
    from .data import ImageDataset, make_transform
    loader = DataLoader(ImageDataset(paths, make_transform(preprocess)), batch_size=batch_size,
                        num_workers=workers, pin_memory=device.type == 'cuda')
    proj = proj.to(device, torch.float32)
    model.eval()
    out = []
    for images in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            features = model.features(images.to(device, non_blocking=True))
        out.append((features.float() @ proj).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def initialize_head(model, state, embeddings, labels, l2=0.1, source='clip_openai_l14'):
    """Set model.head to the folded text-prior head; returns the fitted joint-space weight."""
    proj = state['visual.proj'].double().numpy()  # [1024, 768]
    weight, bias = fit_text_prior(embeddings, labels, text_directions(state, source=source), l2)
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy((proj @ weight).T))
        model.head.bias.copy_(torch.from_numpy(bias))
    return weight, bias
