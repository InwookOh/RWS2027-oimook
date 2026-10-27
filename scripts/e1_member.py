"""Convert a teammate's E1 checkpoint (AttriVision CLIP L/14 + AttriQuery) into our member format.

Uses the E1 code in third_party/attrivision and needs open_clip. The output is an ordinary inference checkpoint (architecture
`e1_clip_l14`, upar/e1.py) that build_submission.py packs with our members; the 52 state-prompt
embeddings are computed here, so submissions need no text tower.

    python scripts/e1_member.py convert \
        --checkpoint outputs/k8s/e1/full/b/checkpoint_epoch_003.pth --output outputs/soups/full/e1-b3.pt
    # check against the E1 code's own predictions on a LODO run (scripts/attrivision_lodo.py export-e1)
    python scripts/e1_member.py check \
        --checkpoint outputs/k8s/e1/lodo-PETA/b6/checkpoint_epoch_003.pth \
        --cache outputs/k8s/e1/lodo-PETA/b6/holdout_predictions_epoch3.npz
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.attributes import ATTRIBUTES  # noqa: E402
from upar.e1 import E1Member  # noqa: E402
from upar.evaluation import evaluate_split  # noqa: E402
from upar.pretrained import CLIP_MEAN, CLIP_STD  # noqa: E402
from upar.results import load_predictions, save_predictions  # noqa: E402
from upar.retrieval import predict_probabilities  # noqa: E402

PREPROCESS = {'height': 224, 'width': 224, 'mean': CLIP_MEAN, 'std': CLIP_STD, 'interpolation': 'bicubic'}


def import_e1(repo):
    """Put the E1 code first on the path. It ships its own `upar` package, so ours is dropped
    from sys.modules (everything this script needs from ours is already imported)."""
    for name in [m for m in sys.modules if m == 'upar' or m.startswith('upar.')]:
        del sys.modules[name]
    sys.path.insert(0, str(Path(repo).expanduser().resolve() / 'examples' / 'task2' / 'sample_code_submission'))
    import attrivision.engine  # noqa: F401  (the E1 code's own import order)
    from attrivision.datasets.attribute_prompts import CategoryPromptMapper
    from attrivision.shared_clip_l14.experiment import _load_shared_model
    return _load_shared_model, CategoryPromptMapper


@torch.no_grad()
def convert(repo, path):
    load_shared, mapper_class = import_e1(repo)
    source, payload = load_shared(path, torch.device('cpu'))
    source.eval()
    mapper = mapper_class(list(payload['attribute_names']))
    member = E1Member((224, 224))
    visual = source.base.clip.visual.state_dict()
    member.proj.copy_(visual.pop('proj'))
    member.visual.load_state_dict(visual, strict=True)
    member.local_head.load_state_dict(source.local_head.state_dict(), strict=True)
    states = source.encode_text(source.base.tokenize(mapper.prompts)).float()
    member.state_features.copy_(torch.nn.functional.normalize(states, dim=-1))
    for category, group in enumerate(mapper.category_indices()):
        member.state_category[list(group)] = category
    state_of = {key: i for i, key in enumerate(mapper.keys)}
    for _, attributes, keys, _ in mapper._MULTI_GROUPS:
        for attribute, key in zip(attributes, keys):
            member.state_attribute[state_of[key]] = ATTRIBUTES.index(attribute)
    for attribute, positive, _ in mapper._BINARY_GROUPS:
        member.state_attribute[state_of[positive]] = ATTRIBUTES.index(attribute)
    assert sorted(member.state_attribute[member.state_attribute >= 0].tolist()) == list(range(40))
    config = {'preprocess': PREPROCESS, 'e1_source': str(Path(path).resolve()), 'e1_epoch': payload.get('epoch'),
              'e1_holdout_domain': payload.get('training_config', {}).get('holdout_domain')}
    return member, {
        'format_version': 1, 'architecture': 'e1_clip_l14', 'attribute_names': list(ATTRIBUTES),
        'preprocess': PREPROCESS, 'model_state': {k: v.detach().cpu() for k, v in member.state_dict().items()},
        'inference_batch_size': 32, 'query_chunk_size': 64, 'epoch': payload.get('epoch'), 'validation_mADM': None,
        'training_config': config, 'weight_source': 'model',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('convert', 'check', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--repo', default=str(Path(__file__).resolve().parents[1] / 'third_party' / 'attrivision'),
                       help='E1 code (default: third_party/attrivision)')
        p.add_argument('--checkpoint', required=True, help='E1-B checkpoint_epoch_NNN.pth')
        if name == 'convert':
            p.add_argument('--output', required=True)
        else:
            p.add_argument('--cache', required=True, help='holdout_predictions_epochN.npz from attrivision_lodo.py export-e1')
            if name == 'export':
                p.add_argument('--out', required=True, help='Held-out cache written with our port')
                p.add_argument('--flip', action='store_true', help='Horizontal-flip TTA')
            p.add_argument('--data-dir', default='data')
            p.add_argument('--batch-size', type=int, default=32)
            p.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    member, checkpoint = convert(args.repo, args.checkpoint)
    if args.command == 'convert':
        target = Path(args.output)
        if target.exists():
            parser.error(f'{target} exists')
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)
        print(f'Wrote {target} (E1 epoch {checkpoint["epoch"]})')
        return
    cache = load_predictions(args.cache)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    member = member.to(device).eval()
    paths = [str(Path(args.data_dir).resolve() / name) for name in cache['image_names']]
    if args.command == 'export':
        probabilities = predict_probabilities(member, paths, PREPROCESS, device, args.batch_size, args.workers,
                                              flip=args.flip, amp=device.type == 'cuda')
        save_predictions(args.out, probabilities, cache['labels'], cache['queries'], cache['ids'],
                         cache['image_names'], {'source': str(Path(args.checkpoint).resolve()), 'flip': args.flip})
        print(f'wrote {args.out}')
        return
    ours = predict_probabilities(member, paths, PREPROCESS, device, args.batch_size, args.workers,
                                 amp=device.type == 'cuda')
    reference = cache['probs'].astype(np.float32)
    difference = np.abs(ours - reference)
    print(f'max |ours - E1 code| = {difference.max():.4f}, mean = {difference.mean():.5f}')
    for name, probabilities in (('E1 code', reference), ('ours', ours)):
        report = evaluate_split(cache['queries'], cache['labels'], cache['ids'], probabilities,
                                cache['image_names'].tolist(), by_domain=False, retrieval='bce')
        print(f'{name:8s} held-out mADM (BCE) {report["metrics"]["mADM"]:.4f}')


if __name__ == '__main__':
    main()
