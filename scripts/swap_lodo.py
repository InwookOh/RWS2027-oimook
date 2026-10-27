"""Score every combination of member versions on the LODO held-out caches with the submitted scoring.

Each --slot is one ensemble member with one or more versions (e.g. the original run and its
train + validation rerun); every combination of versions is scored with the full submission
scoring: per-member label-smoothing correction (eps 0.05, or --unsmooth; members marked @raw left as they are),
calmix (0.5 * BCE of the average + 0.5 * likelihood mixture at T 4) and, with --text-cache, minus
lambda * cos(IRRA image, query-prompt text). Single swaps do not add up, so the joint table is
what decides the submission.

A version `none` drops that slot from the combination (e.g. --slot "new=none,<cache>" scores the
ensemble with and without an added member), and --max-changes K scores only the combinations that
differ from the base (version [0] everywhere) in at most K slots.

--smooth K,W (repeatable, needs --text-cache or --neighbour-cache) also scores every combination with gallery
neighbour smoothing: D <- (1 - W) * D + W * mean of D over the image's K nearest gallery
neighbours by cosine of the IRRA image embeddings (a frame with one confidently wrong attribute is
pulled back by its neighbours, mostly the same person). K,W,mutual uses only mutual neighbours (j among i's K and i among j's K): D <- D + W/K *
sum over i's mutual neighbours of (D_j - D_i), so an image without mutual neighbours (often a person
seen once) stays as it is; rank-based, so it does not depend on the embedding's cosine scale.
--neighbour-cache PATTERN takes the neighbours from other image features instead (an
export_features.py file; the text score still uses --text-cache, and without it the members are
scored alone). --min-positives K scores only queries with at least K gallery images (other images stay
as distractors), as in rescore_lodo.py.

    python scripts/swap_lodo.py \
        --slot "metaclip=outputs/soups/wise-metaclip-b7-e3/{domain}/alpha0.7.npz" \
        --slot "query=outputs/soups/wise-b8-query-e3/{domain}/alpha0.7.npz" \
        --slot "e1=outputs/k8s/e1/lodo-{domain}/b6:3@raw" \
        --slot "dinov2=outputs/soups/wise-dinov2-e4/{domain}/alpha0.8.npz,outputs/soups/wise-dinov2-val-e4/{domain}/alpha0.8.npz" \
        --slot "pe=outputs/soups/wise-pe-e4/{domain}/alpha0.8.npz,outputs/soups/wise-pe-val-e4/{domain}/alpha0.7.npz" \
        --slot "irra=outputs/k8s/irra-metaclip/lodo-{domain}:2@raw" \
        --text-cache "outputs/k8s/irra-metaclip/lodo-{domain}/globals_epoch2.npz" --text-lambda 5
"""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
from itertools import product
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from ensemble_lodo import cache_path  # noqa: E402
from rescore_lodo import calibrate, calmix_distance, text_similarity  # noqa: E402
from upar.data import DOMAINS, load_split, select_domains  # noqa: E402
from upar.evaluation import evaluate_split  # noqa: E402
from upar.results import load_predictions  # noqa: E402

DATA = {}  # per worker: domain -> loaded caches (filled by the pool initializer)


def neighbours(image, k, chunk=2048):
    """Indices of each gallery image's k nearest other images by cosine, nearest first."""
    image = image / np.linalg.norm(image, axis=1, keepdims=True).clip(min=1e-12)
    result = np.zeros((len(image), k), dtype=np.int64)
    for start in range(0, len(image), chunk):
        sims = image[start:start + chunk] @ image.T
        sims[np.arange(len(sims)), start + np.arange(len(sims))] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, top, 1), axis=1)
        result[start:start + chunk] = np.take_along_axis(top, order, 1)
    return result


def load(domain, slots, text_cache, prior, unsmooth=0.05, max_k=0, min_positives=1, neighbour_cache=None):
    versions = {}
    names = None
    for slot, choices in enumerate(slots):
        for choice, (member, raw) in enumerate(choices):
            if member is None:
                versions[slot, choice] = None
                continue
            cache = load_predictions(cache_path(member, domain))
            if names is None:
                names, reference = cache['image_names'], cache
            elif not np.array_equal(cache['image_names'], names):
                raise ValueError(f'{domain}: gallery order differs for {member}')
            probs = cache['probs'].astype(np.float64)
            versions[slot, choice] = probs if raw or unsmooth == 0 else calibrate(probs, prior, unsmooth, 0.0)
    data = {key: reference[key] for key in ('queries', 'labels', 'ids', 'image_names')}
    data['versions'] = versions
    counts = np.bincount(data['ids'], minlength=len(data['queries']))
    keep = counts >= min_positives
    data['eval_ids'] = np.where(keep[data['ids']], data['ids'], np.flatnonzero(keep)[0])
    if text_cache:
        cache = np.load(cache_path(text_cache, domain))
        if not np.array_equal(cache['image_names'], names):
            raise ValueError(f'{domain}: IRRA embedding gallery order differs')
        data['image'] = cache['image'].astype(np.float64)
        data['text'] = cache['text'].astype(np.float64)
        data['text_row'] = {tuple(int(x) for x in q): i for i, q in enumerate(cache['queries'])}
    if max_k:
        if neighbour_cache:
            features = np.load(cache_path(neighbour_cache, domain))
            if not np.array_equal(features['image_names'], names):
                raise ValueError(f'{domain}: neighbour feature gallery order differs')
            embedding = features['image'].astype(np.float64)
        else:
            embedding = data['image']
        data['neighbours'] = neighbours(embedding, max_k)
    return data


def init(domains, slots, text_cache, priors, unsmooth, max_k, min_positives, neighbour_cache):
    for domain in domains:
        DATA[domain] = load(domain, slots, text_cache, priors[domain], unsmooth, max_k, min_positives,
                            neighbour_cache)


def score(task):
    combo, domain, lam, smooth = task
    data = DATA[domain]
    members = np.stack([data['versions'][slot, choice] for slot, choice in enumerate(combo)
                        if data['versions'][slot, choice] is not None])
    count = len(members)
    packed = np.concatenate([members.mean(0)[:, None], members.transpose(1, 0, 2)], 1)
    packed = packed.reshape(len(members[0]), -1).astype(np.float32)

    def method(queries, p):
        d = calmix_distance(queries, p, count)
        d = d - lam * text_similarity(queries, data) if 'text' in data else d
        if smooth:
            k, w, mutual = smooth
            near = data['neighbours'][:, :k]
            if mutual:
                keep = (near[near] == np.arange(len(near))[:, None, None]).any(-1)  # G x k
                d = d + w / k * ((d[:, near] - d[:, :, None]) * keep).sum(-1)
            else:
                d = (1 - w) * d + w * d[:, near].mean(-1)
        return d
    with contextlib.redirect_stdout(io.StringIO()):
        report = evaluate_split(data['queries'], data['labels'], data['eval_ids'], packed,
                                data['image_names'].tolist(), by_domain=False, retrieval=method)
    return combo, domain, smooth, report['metrics']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--slot', action='append', required=True,
                        help='NAME=VERSION[,VERSION...], each a cache pattern with {domain}, @raw to skip calibration')
    parser.add_argument('--text-cache', help='IRRA embeddings (.npz pattern with {domain})')
    parser.add_argument('--text-lambda', type=float, default=5.0)
    parser.add_argument('--unsmooth', type=float, default=0.05,
                        help='Label-smoothing correction of the smoothed members (0: a submission without --unsmooth)')
    parser.add_argument('--domains', nargs='+', default=['Market1501', 'PETA'])
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--smooth', action='append', default=[], metavar='K,W[,mutual]',
                        help='Also score with gallery neighbour smoothing (K neighbours, weight W); needs --text-cache')
    parser.add_argument('--neighbour-cache', help='Image features for the --smooth neighbours (.npz pattern with '
                        '{domain}, from export_features.py); default: the IRRA image embeddings of --text-cache')
    parser.add_argument('--min-positives', type=int, default=1,
                        help='Score only queries with at least this many gallery images')
    parser.add_argument('--max-changes', type=int,
                        help='Only combinations differing from the base in at most this many slots')
    args = parser.parse_args()
    smooths = [None]
    for text in args.smooth:
        k, w, *mode = text.split(',')
        if mode not in ([], ['mutual']):
            parser.error(f'--smooth {text}: expected K,W or K,W,mutual')
        smooths.append((int(k), float(w), bool(mode)))
    if args.neighbour_cache and not args.smooth:
        parser.error('--neighbour-cache needs --smooth')
    if args.smooth and not (args.text_cache or args.neighbour_cache):
        parser.error('--smooth needs --text-cache (its IRRA image embeddings define the neighbours) or --neighbour-cache')
    max_k = max([s[0] for s in smooths if s] or [0])
    names, slots = [], []
    for text in args.slot:
        name, _, versions = text.partition('=')
        choices = []
        for version in versions.split(','):
            raw = version.endswith('@raw')
            choices.append((None, False) if version == 'none' else (version[:-4] if raw else version, raw))
        names.append(name)
        slots.append(choices)
    missing = sorted({m for choices in slots for m, _ in choices for d in args.domains
                      if m is not None and not cache_path(m, d).exists()})
    if missing:
        sys.exit('missing caches:\n  ' + '\n  '.join(missing))
    train_names, train_labels, _, train_ids = load_split(args.data_dir, 'train')
    priors = {}
    for domain in args.domains:
        _, seen, _ = select_domains(train_names, train_labels, train_ids, [d for d in DOMAINS if d != domain])
        priors[domain] = seen.mean(0).astype(np.float64)
    combos = [c for c in product(*[range(len(c)) for c in slots])
              if args.max_changes is None or sum(i != 0 for i in c) <= args.max_changes]
    tasks = [(c, d, args.text_lambda, s) for c in combos for s in smooths for d in args.domains]
    results = {}
    with ProcessPoolExecutor(args.jobs, initializer=init,
                             initargs=(args.domains, slots, args.text_cache, priors, args.unsmooth,
                                       max_k, args.min_positives, args.neighbour_cache)) as executor:
        for combo, domain, smooth, metrics in executor.map(score, tasks):
            results[combo, smooth, domain] = metrics
    for name, choices in zip(names, slots):
        print(f'{name}: ' + '  '.join(f'[{i}] {m or "none"}' + (' (raw)' if raw else '') for i, (m, raw) in enumerate(choices)))
    base = tuple(0 for _ in slots)
    rows = []
    for combo in combos:
        for smooth in smooths:
            per = [results[combo, smooth, d] for d in args.domains]
            rows.append((np.mean([m['mADM'] for m in per]), combo, smooth, per))
    base_per = [results[base, None, d]['mADM'] for d in args.domains]
    print(f'\nversion index per slot ({", ".join(names)}); [0] everywhere is the base; text lambda '
          f'{args.text_lambda if args.text_cache else "none"}'
          + (f'; queries with >= {args.min_positives} images' if args.min_positives > 1 else '')
          + ('; smooth = K neighbours, weight' if args.smooth else ''))
    print(f'{"versions":20s} ' + ' '.join(f'{d:>10s}' for d in args.domains)
          + '       mADM    gain     mAP  Rank-1    mINP  all splits up')
    for madm, combo, smooth, per in sorted(rows, key=lambda r: -r[0]):
        up = all(m['mADM'] >= b for m, b in zip(per, base_per))
        label = ''.join(str(c) for c in combo) + (f' s{smooth[0]},{smooth[1]:g}' + ('m' if smooth[2] else '') if smooth else '')
        print(f'{label:20s} ' + ' '.join(f'{m["mADM"]:10.4f}' for m in per)
              + f'   {madm:.4f} {madm - np.mean(base_per):+.4f}  '
              + ' '.join(f'{np.mean([m[k] for m in per]):.4f}' for k in ('mAP', 'Rank-1', 'mINP'))
              + ('  yes' if up else ''))


if __name__ == '__main__':
    main()
