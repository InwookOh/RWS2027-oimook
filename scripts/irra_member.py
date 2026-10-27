"""Convert a teammate's IRRA-QB FrozenMultiScaleLocalAttr checkpoint into our member format.

The checkpoint is the plain 532-tensor state_dict written by the IRRA code in third_party/
(epoch_002.pth). The output is an inference checkpoint (architecture `irra_clip_l14`,
upar/irra.py) for build_submission.py; it keeps the image tower, the multiscale fusion and the
attribute branch (the text tower and the masked-prediction modules are dropped).

    python scripts/irra_member.py convert --checkpoint outputs/k8s/irra-metaclip/full/epoch_002.pth \
        --output outputs/soups/full/irra-e2.pt
    # compare with the original code on gallery images (its repository checked out)
    python scripts/irra_member.py check \
        --checkpoint outputs/k8s/irra-metaclip/full/epoch_002.pth --images 64
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.attributes import ATTRIBUTES  # noqa: E402
from upar.irra import IRRAMember  # noqa: E402
from upar.pretrained import CLIP_MEAN, CLIP_STD  # noqa: E402

PREPROCESS = {'height': 378, 'width': 126, 'mean': CLIP_MEAN, 'std': CLIP_STD}


def read_state(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not any(k.startswith('base_model.visual.') for k in state):
        raise ValueError(f'{path}: expected the plain FrozenMultiScaleLocalAttr state_dict')
    return state


def convert(path, text=False):
    state = read_state(path)
    mapped = {}
    if text:  # the fine-tuned text tower, for the CLIP image-text score
        from upar.clip_text import openai_text_state
        tower = {k[len('base_model.'):]: v for k, v in state.items() if k.startswith('base_model.')
                 and not k.startswith('base_model.visual.')}
        mapped.update({'text.' + k: v for k, v in openai_text_state(tower).items()})
    for key, value in state.items():
        if key == 'base_model.visual.proj':
            mapped['proj'] = value
        elif key.startswith('base_model.visual.'):
            mapped['visual.' + key[len('base_model.visual.'):]] = value
        elif key.startswith(('multiscale.', 'local_branch.')):
            mapped[key] = value
    member = IRRAMember((PREPROCESS['height'], PREPROCESS['width']), text=text)
    member.load_state_dict({k: v.float() for k, v in mapped.items()}, strict=True)
    config = {'preprocess': PREPROCESS, 'irra_source': str(Path(path).resolve()), 'irra_text': text}
    extra = {}
    if text:  # gzipped CLIP BPE vocabulary for the query prompts (no download at inference)
        from upar.clip_text import bpe_path
        extra['irra_bpe'] = torch.tensor(list(Path(bpe_path()).read_bytes()), dtype=torch.uint8)
    return member, {**extra,
        'format_version': 1, 'architecture': 'irra_clip_l14', 'attribute_names': list(ATTRIBUTES),
        'preprocess': PREPROCESS, 'model_state': {k: v.detach().cpu() for k, v in member.state_dict().items()},
        'inference_batch_size': 32, 'query_chunk_size': 64, 'epoch': None, 'validation_mADM': None,
        'training_config': config, 'weight_source': 'model',
    }


@torch.no_grad()
def check(args):
    member, checkpoint = convert(args.checkpoint, text=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    member = member.to(device).eval()
    member.keep_global = True
    paths = sorted(str(p) for d in ('Market1501', 'PETA', 'PA100k')
                   for p in (Path(args.data_dir) / d).rglob('*') if p.suffix.lower() in ('.jpg', '.png', '.jpeg'))
    paths = paths[::max(1, len(paths) // args.images)][:args.images]
    from upar.data import load_split
    from upar.retrieval import predict_probabilities
    ours = predict_probabilities(member, paths, PREPROCESS, device, 16, 0)
    queries = load_split(args.data_dir, 'val')[2][:args.queries].astype(int)
    our_text = member.query_features(queries, bytes(checkpoint['irra_bpe'].tolist())).cpu()
    # The original code ships its own `upar` package: drop ours from sys.modules first.
    for name in [m for m in sys.modules if m == 'upar' or m.startswith('upar.')]:
        del sys.modules[name]
    sys.path.insert(0, str(Path(args.repo).resolve()))
    import inference as original
    from upar.images import image_transform, read_rgb
    from upar.prompts import encode_query
    from upar.tokenizer import PromptTokenizer
    model = original.load_model(args.checkpoint, device)
    transform = image_transform()
    logits, globals_ours, globals_ref = [], [], []
    for start in range(0, len(paths), 16):
        images = torch.stack([transform(read_rgb(p)) for p in paths[start:start + 16]]).to(device)
        features, batch_logits = model.encode_image_and_attributes(images)
        logits.append(batch_logits.float().cpu())
        globals_ref.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        member(images)  # the same preprocessed batch through our port
        globals_ours.append(member.last_global.cpu())
    reference = torch.cat(logits).sigmoid().numpy()
    difference = np.abs(ours - reference)
    print(f'{len(paths)} images: attribute probabilities max |ours - original| = {difference.max():.5f}, '
          f'mean = {difference.mean():.6f}')
    image_cos = (torch.cat(globals_ours) * torch.cat(globals_ref)).sum(1)
    print(f'global image embedding cosine(ours, original): min {image_cos.min():.6f}')
    tokenizer = PromptTokenizer()
    tokens = torch.tensor([encode_query(tuple(int(x) for x in q), tokenizer).token_ids for q in queries]).to(device)
    ref_text = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1).cpu()
    text_cos = (our_text * ref_text).sum(1)
    print(f'{len(queries)} queries: text embedding cosine(ours, original): min {text_cos.min():.6f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('convert')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--output', required=True)
    c.add_argument('--text', action='store_true', help='Keep the text tower for the image-text score')
    k = sub.add_parser('check')
    k.add_argument('--repo', default=str(Path(__file__).resolve().parents[1] / 'third_party' / 'irra'),
                   help='IRRA code (default: third_party/irra)')
    k.add_argument('--checkpoint', required=True)
    k.add_argument('--data-dir', default='data')
    k.add_argument('--images', type=int, default=64)
    k.add_argument('--queries', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'convert':
        target = Path(args.output)
        if target.exists():
            parser.error(f'{target} exists')
        _, checkpoint = convert(args.checkpoint, text=args.text)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)
        print(f'Wrote {target}')
    else:
        check(args)


if __name__ == '__main__':
    main()
