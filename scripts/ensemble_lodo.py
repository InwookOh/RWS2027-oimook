"""Score ensembles of LODO runs on the held-out domains, without retraining.

Each member is PATTERN:EPOCH, where PATTERN contains {domain} and points to a LODO run
folder holding holdout_predictions_epoch<EPOCH>.npz, or a .npz path pattern with {domain}
(e.g. the WiSE-FT caches of scripts/soup_lodo.py). Member probabilities are averaged
per split, then retrieval is scored on each held-out gallery and averaged over splits.
`--weights 0.25,0.25,0.5 0.5,0.25,0.25 ...` scores the full member set with each weight vector
(normalized to sum to one) instead of uniform subsets. `--fusion` picks how members combine:
prob (average probabilities, the submission default), logit (average logits), distance
(average each member's retrieval distances; for BCE a product of the members' likelihoods)
or rank (average each member's gallery ranks per query).

    python scripts/ensemble_lodo.py \
        outputs/B1/lodo-amp-k8s/{domain}:5 \
        outputs/k8s/b1-lpft/lodo-{domain}:6 \
        outputs/k8s/clip-lr2e-5/lodo-{domain}:6 --subsets
"""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from upar.data import DOMAINS  # noqa: E402
from upar.evaluation import evaluate_split  # noqa: E402
from upar.retrieval import check_retrieval, query_distances  # noqa: E402
from upar.results import load_predictions  # noqa: E402


def cache_path(member, domain):
    if member.endswith('.npz'):  # a cache file pattern, e.g. outputs/soups/<run>/{domain}/alpha0.8.npz
        return ROOT / member.format(domain=domain)
    pattern, tag = member.rsplit(':', 1)  # :4 for epoch 4 (an E1 or IRRA tag such as :b6 is kept as is)
    tag = f'epoch{int(tag)}' if tag.isdigit() else tag
    return ROOT / pattern.format(domain=domain) / f'holdout_predictions_{tag}.npz'


def score(task):
    members, domain, retrieval, vector, fusion = task
    caches = [load_predictions(cache_path(m, domain)) for m in members]
    names = caches[0]['image_names']
    for member, cache in zip(members, caches):
        if not np.array_equal(cache['image_names'], names):
            raise ValueError(f'{member}: {domain} gallery order differs from {members[0]}')
    weights = np.full(len(caches), 1 / len(caches)) if vector is None else np.asarray(vector) / sum(vector)
    stacked = np.stack([c['probs'] for c in caches]).astype(np.float64)  # [M, G, 40]
    method = retrieval
    if fusion == 'prob':
        probabilities = np.tensordot(weights, stacked, axes=1).astype(np.float32)
    elif fusion == 'logit':
        p = np.clip(stacked, 1e-6, 1 - 1e-6)
        probabilities = (1 / (1 + np.exp(-np.tensordot(weights, np.log(p) - np.log1p(-p), axes=1)))).astype(np.float32)
    else:
        # Pack the members side by side ([G, M*40]) and fuse their per-member distances.
        count = len(caches)
        probabilities = stacked.transpose(1, 0, 2).reshape(stacked.shape[1], -1).astype(np.float32)

        def method(queries, packed):
            parts = packed.reshape(len(packed), count, 40)
            total = 0.0
            for m in range(count):
                d = query_distances(queries, parts[:, m], retrieval).astype(np.float64)
                if fusion == 'rank':
                    d = np.argsort(np.argsort(d, axis=1, kind='stable'), axis=1, kind='stable').astype(np.float64)
                total = total + weights[m] * d
            return total
    first = caches[0]
    with contextlib.redirect_stdout(io.StringIO()):
        report = evaluate_split(first['queries'], first['labels'], first['ids'], probabilities,
                                names.tolist(), by_domain=False, retrieval=method)
    return (members, vector, fusion), domain, report['metrics']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('members', nargs='+', help='PATTERN_WITH_{domain}:EPOCH')
    parser.add_argument('--domains', nargs='+', default=list(DOMAINS), choices=DOMAINS)
    parser.add_argument('--retrieval', nargs='+', default=['bce'], type=check_retrieval,
                        help="l1 or bce; several values are compared (full member set only)")
    parser.add_argument('--subsets', action='store_true', help='Score every non-empty subset of members')
    parser.add_argument('--weights', nargs='+', help='Comma-separated weight vectors, one weight per member')
    parser.add_argument('--fusion', nargs='+', default=['prob'], choices=('prob', 'logit', 'distance', 'rank'),
                        help='How members combine; several values are compared (full member set only)')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    vectors = None
    if args.weights:
        vectors = [tuple(float(w) for w in v.split(',')) for v in args.weights]
        if args.subsets or any(len(v) != len(args.members) or min(v) < 0 or sum(v) <= 0 for v in vectors):
            parser.error('--weights needs one nonnegative weight per member (not with --subsets)')
        vectors = [None] + vectors
    for member in args.members:
        if '{domain}' not in member or (':' not in member and not member.endswith('.npz')):
            parser.error(f'{member}: expected PATTERN_WITH_{{domain}}:EPOCH or a .npz pattern with {{domain}}')
        for domain in args.domains:
            if not cache_path(member, domain).exists():
                parser.error(f'missing {cache_path(member, domain)}')
    sizes = range(1, len(args.members) + 1) if args.subsets else [1, len(args.members)]
    groups = list(dict.fromkeys(c for k in sizes for c in combinations(args.members, k)))
    if vectors:
        groups = [tuple(args.members)]
    if (len(args.fusion) > 1 or len(args.retrieval) > 1) and args.subsets:
        parser.error('compare several --fusion or --retrieval values on the full member set (without --subsets)')
    if len(args.retrieval) > 1:
        groups = [tuple(args.members)]
    keys = [(group, w, f, r) for group in groups for w in (vectors or [None])
            for f in (args.fusion if len(group) > 1 else ['prob']) for r in args.retrieval]
    tasks = [(group, domain, r, w, f) for group, w, f, r in keys for domain in args.domains]
    results = {}
    with ProcessPoolExecutor(args.jobs) as executor:
        for (key, task) in zip(executor.map(score, tasks), tasks):
            (group, w, f), domain, metrics = key
            results[(group, w, f, task[2]), domain] = metrics
    labels = {m: chr(ord('A') + i) for i, m in enumerate(args.members)}
    for member, label in labels.items():
        print(f'{label} = {member}')
    several = len(args.retrieval) > 1
    print(f'\nretrieval={",".join(args.retrieval)}, fusion={",".join(args.fusion)}; held-out mADM per split, '
          f'then mean mADM / mean mAP' + (' / mean Rank-1' if several else ''))
    print(f'{"members":16s} ' + ' '.join(f'{d:>10s}' for d in args.domains) + '   mean mADM  mean mAP'
          + ('  Rank-1' if several else ''))
    rows = []
    for key in keys:
        madm = [results[key, d]['mADM'] for d in args.domains]
        mean_map = np.mean([results[key, d]['mAP'] for d in args.domains])
        rank1 = np.mean([results[key, d]['Rank-1'] for d in args.domains])
        group, w, fusion, retrieval = key
        name = '+'.join(labels[m] for m in group)
        if vectors:
            name = 'uniform' if w is None else ','.join(f'{x:g}' for x in w)
        if len(args.fusion) > 1 and len(group) > 1:
            name = f'{name} {fusion}'
        if several:
            name = retrieval
        rows.append((np.mean(madm), len(group), name, madm, mean_map, rank1))
    for mean_madm, _, name, madm, mean_map, rank1 in sorted(rows, key=lambda r: (r[1], -r[0])):
        print(f'{name:16s} ' + ' '.join(f'{v:10.4f}' for v in madm) + f'   {mean_madm:9.4f}  {mean_map:8.4f}'
              + (f'  {rank1:6.4f}' if several else ''))


if __name__ == '__main__':
    main()
