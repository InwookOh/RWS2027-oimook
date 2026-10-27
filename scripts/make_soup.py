"""Write one checkpoint that mixes the weights of several, for build_submission.py.

Members are averaged uniformly; with --base the result is
(1 - alpha) * base + alpha * soup (WiSE-FT). All checkpoints must come from the same
pretrained initialization (same backbone, input size and LP-FT start).

    python scripts/make_soup.py outputs/k8s/b4-lr5e-6/full/epoch4.pt \
        --base outputs/k8s/b4-lr5e-6/full/epoch1.pt --alpha 0.8 --output outputs/soups/b4lr5e-6-wise0.8.pt
"""
import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.soup import load_checkpoint, mix_checkpoints  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('members', nargs='+', help='Fine-tuned epoch<N>.pt checkpoints, averaged uniformly')
    parser.add_argument('--base', nargs='+', help='Checkpoint(s), averaged, interpolated with the soup (e.g. the linear-probe epoch1.pt)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight of the soup against --base')
    parser.add_argument('--output', required=True)
    parser.add_argument('--domain-soup', action='store_true',
                        help='Allow leave-one-domain-out checkpoints when the fine-tuned members hold out each '
                             'domain exactly once (a soup of the three LODO runs; the result counts as all-domain)')
    args = parser.parse_args()
    if args.base is None and args.alpha != 1.0:
        parser.error('--alpha needs --base')
    target = Path(args.output)
    if target.exists():
        parser.error(f'{target} already exists; choose a new output path')
    paths = (args.base or []) + args.members
    checkpoints = [load_checkpoint(p) for p in paths]
    holdouts = [c.get('training_config', {}).get('holdout_domain') for c in checkpoints]
    if args.domain_soup:
        from upar.data import DOMAINS
        members = holdouts[len(args.base or []):]
        if sorted(members) != sorted(DOMAINS):
            parser.error(f'--domain-soup needs one fine-tuned member per held-out domain {DOMAINS}, got {members}')
        if args.base:
            bases = holdouts[:len(args.base)]
            if sorted(bases) != sorted(DOMAINS):
                parser.error(f'--domain-soup needs one base per held-out domain, got {bases}')
    else:
        for path, holdout in zip(paths, holdouts):
            if holdout:
                parser.error(f'{path}: leave-one-domain-out checkpoint; mix all-domain runs for submissions '
                             '(or --domain-soup for a soup of all three LODO runs)')
    if args.base:
        weights = [(1 - args.alpha) / len(args.base)] * len(args.base) + [args.alpha / len(args.members)] * len(args.members)
    else:
        weights = [1 / len(args.members)] * len(args.members)
    mixed = mix_checkpoints(checkpoints, weights, paths)
    if args.domain_soup:
        # Every domain is seen by two of the three members: treat the soup as an all-domain model.
        mixed['training_config'] = {**mixed.get('training_config', {}), 'holdout_domain': None,
                                    'domain_soup': sorted(set(holdouts))}
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mixed, target)
    print(f'Wrote {target}: weights {dict(zip(paths, (round(w, 4) for w in weights)))}')


if __name__ == '__main__':
    main()
