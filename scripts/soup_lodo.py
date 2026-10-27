"""Score weight soups and WiSE-FT interpolations of LODO runs on the held-out domains.

Members are PATTERN:EPOCH as in ensemble_lodo.py; their epoch<EPOCH>.pt weights are
averaged into one model. With --base, that soup is interpolated with the base checkpoint
for every --alpha (WiSE-FT): weights = (1 - alpha) * base + alpha * soup. The LP-FT
linear-probe epoch (epoch 1: pretrained backbone, trained head) is the natural base.
Each mixed model predicts its held-out gallery (as train.py does, without TTA); caches
are saved under --output-dir/<domain>/ for later ensembling.

    python scripts/soup_lodo.py outputs/k8s/b4-lr5e-6/lodo-{domain}:4 \
        --base outputs/k8s/b4-lr5e-6/lodo-{domain}:1 --alphas 0.5 0.7 0.8 0.9 1
    python scripts/soup_lodo.py outputs/k8s/b4/lodo-{domain}:4 outputs/k8s/b4-lr5e-6/lodo-{domain}:4
"""
import argparse
import contextlib
import io
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.data import DOMAINS  # noqa: E402
from upar.evaluation import evaluate_split  # noqa: E402
from upar.results import load_predictions, save_predictions  # noqa: E402
from upar.retrieval import predict_probabilities  # noqa: E402
from upar.soup import load_checkpoint, mix_checkpoints, model_from_checkpoint  # noqa: E402


def run_path(member, domain):
    """(run folder, checkpoint name) for PATTERN:4 (epoch4.pt)."""
    pattern, tag = member.rsplit(':', 1)
    return ROOT / pattern.format(domain=domain), f'epoch{int(tag)}'


def cache_name(tag):
    return f'holdout_predictions_{tag}.npz'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('members', nargs='+', help='PATTERN_WITH_{domain}:EPOCH, averaged uniformly')
    parser.add_argument('--base', nargs='+', help='PATTERN_WITH_{domain}:EPOCH (several are averaged) interpolated with the soup (WiSE-FT)')
    parser.add_argument('--alphas', nargs='+', type=float, default=[1.0], help='Weight of the soup against --base')
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument('--retrieval', default='bce', choices=('l1', 'bce'))
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs/soups/latest')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8, help='CPU threads (parallel jobs otherwise oversubscribe the CPU)')
    parser.add_argument('--flip', action='store_true', help='Horizontal-flip TTA (average with the mirrored image)')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    specs = args.members + (args.base or [])
    for spec in specs:
        if '{domain}' not in spec or ':' not in spec:
            parser.error(f'{spec}: expected PATTERN_WITH_{{domain}}:EPOCH')
        for domain in args.domains:
            folder, tag = run_path(spec, domain)
            if not (folder / f'{tag}.pt').exists():
                parser.error(f'missing {folder / f"{tag}.pt"}')
    if not args.base and args.alphas != [1.0]:
        parser.error('--alphas needs --base')
    data_root = Path(args.data_dir).resolve()
    results = {}
    for domain in args.domains:
        # Gallery, labels and queries come from the first member's held-out prediction cache.
        folder, tag = run_path(args.members[0], domain)
        reference = load_predictions(folder / cache_name(tag))
        names = reference['image_names'].tolist()
        paths = [str(data_root / name) for name in names]
        members = [load_checkpoint(run_path(m, domain)[0] / f'{run_path(m, domain)[1]}.pt') for m in args.members]
        bases = [load_checkpoint(run_path(b, domain)[0] / f'{run_path(b, domain)[1]}.pt') for b in (args.base or [])]
        for alpha in args.alphas:
            checkpoints = bases + members
            if bases:
                weights = [(1 - alpha) / len(bases)] * len(bases) + [alpha / len(members)] * len(members)
            else:
                weights = [1 / len(members)] * len(members)
            mixed = mix_checkpoints(checkpoints, weights, specs)
            model = model_from_checkpoint(mixed, args.device)
            with contextlib.redirect_stdout(io.StringIO()):
                probabilities = predict_probabilities(model, paths, mixed['preprocess'], args.device,
                                                      args.batch_size, args.workers, flip=args.flip)
                report = evaluate_split(reference['queries'], reference['labels'], reference['ids'], probabilities,
                                        names, by_domain=False, retrieval=args.retrieval)
            del model
            save_predictions(Path(args.output_dir) / domain / f'alpha{alpha:g}.npz', probabilities, reference['labels'],
                             reference['queries'], reference['ids'], names,
                             {'members': args.members, 'base': args.base, 'alpha': alpha, 'domain': domain})
            results[domain, alpha] = report['metrics']
            print(f'{domain:10s} alpha={alpha:g}  mADM {report["metrics"]["mADM"]:.4f}  '
                  f'mAP {report["metrics"]["mAP"]:.4f}', flush=True)
    print(f'\nretrieval={args.retrieval}; members={args.members}; base={args.base}')
    print(f'{"alpha":>6s} ' + ' '.join(f'{d:>10s}' for d in args.domains) + '   mean mADM  mean mAP')
    for alpha in args.alphas:
        madm = [results[d, alpha]['mADM'] for d in args.domains]
        mean_map = np.mean([results[d, alpha]['mAP'] for d in args.domains])
        print(f'{alpha:6g} ' + ' '.join(f'{v:10.4f}' for v in madm) + f'   {np.mean(madm):9.4f}  {mean_map:8.4f}')


if __name__ == '__main__':
    main()
