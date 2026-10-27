"""Rescore cached held-out predictions of an ensemble with alternative scoring rules (CPU only).

Members are given as in ensemble_lodo.py. Variants (each scored on every held-out domain):

- base: average the members' probabilities, BCE retrieval (the submission)
- calib(eps,gamma): per member, undo part of the training distortions before averaging:
  p' = sigmoid(logit(clip((p - eps/2) / (1 - eps))) - gamma * (1 - 2 * pi_a)), where eps undoes
  label smoothing (training used 0.1) and gamma the DeepMAR weights exp(1 - pi) / exp(pi)
  (pi_a = positive rate of attribute a in the run's training domains). Members listed in
  --raw (e.g. a converted member trained without DeepMAR) are left unchanged.
- mix(T,w): complete-query likelihood mixture. D_m = member m's BCE distance,
  D_T = -T log mean_m exp(-D_m / T); the score is (1 - w) * base + w * D_T.
- calmix(eps,gamma,T,w): calib then mix, on the calibrated member probabilities (--combo).
- rawbase(eps,gamma,T,w): as calmix, but the BCE term uses the uncalibrated average (--rawbase):
  calibration then only sharpens the mixture term.
- On top of calmix (eps 0.05, T 4, w 0.5; members calibrated as in calib):
  gtemp(tc,th): the BCE term uses sigmoid(logit(P) / t) on the averaged probabilities, t = tc for
    clothing colours, th for hair length, bag and glasses, 1 otherwise (--gtemp)
  gweight(wc,wh): the BCE term weights those groups' attribute losses by wc / wh (--gweight)
  sub(Ts,eta): the mixture term becomes (1 - eta) * D_4 + eta * M_Ts over the BCE distances of
    the leave-one-member-out averages (--sub)
  block(mode,eta): the mixture term becomes (1 - eta) * D_4 + eta * sum over the 12 attribute
    categories of the member mixture of each category's distance, at T = 4 (mode 0) or
    4 * |category| / 40 (mode 1) (--block)
- irratext(lam): calmix distance - lam * cos(IRRA image, IRRA query-prompt text), with the
  embeddings from scripts/irra_lodo.py export --globals (--irra-text-cache, --irratext)
- textfuse(mode,v): other fusions of the same calmix distance D and cosine C, per query over the
  gallery (--textfuse): mode 0, (1 - v) * z(D) - v * z(C) (the teammate's z-score fusion, v = 0.24);
  mode 1, D - v * C / std(C) (only the cosine rescaled per query); mode 2, (1 - v) * r(D) + v * r(-C)
  with r the rank divided by the gallery size (rank fusion)
- nearmiss(mode,a,b), on calmix - 5 * cos (text term only with --irra-text-cache) (--nearmiss):
  mode 0 = oracle bound: exact matches keep their positions, every other image is moved into the
  non-match positions by its true agreement (the ADM reachable at the current AP); mode 1 = sort by
  (floor(D / a), expected mismatches, D); mode 2 = D - min(b, 0.5 * log(1 + a * sum_a (1 - r_a) / r_a))
  with r_a the averaged probability of the query's value (exact plus one-error likelihood)
- --min-positives K scores only queries with at least K gallery images (the hidden test has about 33
  per query; held-out Market1501 17, PETA 4 with 61% single-image queries).
- cap(c): sum over attributes of min(-log r_a, c), r_a = probability of the query's value,
  which limits the penalty of one confidently wrong attribute.

    python scripts/rescore_lodo.py "outputs/soups/wise-metaclip-b7-e3/{domain}/alpha0.7.npz" \
        "outputs/soups/wise-b8-query-e3/{domain}/alpha0.7.npz" outputs/k8s/e1/lodo-{domain}/b6:3 \
        "outputs/soups/wise-dinov2-e4/{domain}/alpha0.8.npz" --raw C
"""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor
import io
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from ensemble_lodo import cache_path  # noqa: E402
from upar.data import DOMAINS, load_split, select_domains  # noqa: E402
from upar.evaluation import evaluate_split  # noqa: E402
from upar.results import load_predictions  # noqa: E402
from upar.retrieval import bce_distances, binary_matrix  # noqa: E402

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def calibrate(p, prior, eps, gamma):
    s = np.clip((p - eps / 2) / (1 - eps), EPS, 1 - EPS)
    return 1 / (1 + np.exp(-(logit(s) - gamma * (1 - 2 * prior)[None, :])))


COLOURS = list(range(8, 20)) + list(range(21, 33))
HAIR_BAG_GLASSES = [4, 5, 6, 36, 37, 38]
CATEGORIES = ((0, 1, 2), (3,), (4, 5, 6), (7,), tuple(range(8, 20)), (20,), tuple(range(21, 33)),
              (33, 34), (35,), (36,), (37, 38), (39,))


def weighted_bce(queries, probabilities, weights=None, columns=None):
    """sum_a w_a * BCE of the query value (optionally over a subset of attribute columns)."""
    q = binary_matrix(queries, 'queries').astype(np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1 - EPS)
    on, off = -np.log(p), -np.log1p(-p)
    if weights is not None:
        on, off = on * weights, off * weights
    if columns is not None:
        q, on, off = q[:, columns], on[:, columns], off[:, columns]
    return off.sum(1)[None, :] + q @ (on - off).T


def soft_min(distances, temperature):
    """-T log mean exp(-d / T) over the first axis."""
    low = distances.min(0)
    return low - temperature * np.log(np.mean(np.exp(-(distances - low) / temperature), axis=0))


def calmix_distance(queries, packed, count, temperature=4.0, weight=0.5):
    parts = packed.reshape(len(packed), count + 1, 40).astype(np.float64)
    mixture = soft_min(np.stack([weighted_bce(queries, parts[:, 1 + m]) for m in range(count)]), temperature)
    return (1 - weight) * weighted_bce(queries, parts[:, 0]) + weight * mixture


def text_similarity(queries, data):
    rows = [data['text_row'][tuple(int(x) for x in q)] for q in np.asarray(queries)]
    return data['text'][rows] @ data['image'].T


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True).clip(min=1e-9)


def ranks(x):
    return x.argsort(1).argsort(1) / x.shape[1]


def capped_distances(queries, probabilities, cap):
    queries = binary_matrix(queries, 'queries')
    p = np.clip(np.asarray(probabilities, dtype=np.float64), EPS, 1 - EPS)
    on, off = np.minimum(-np.log(p), cap), np.minimum(-np.log1p(-p), cap)  # [G, 40]
    return off.sum(1)[None, :] + queries @ (on - off).T


def score(task):
    variant, domain, data = task
    kind, params = variant
    members = data['members']  # [M, G, 40]
    mean = members.mean(0)
    if kind in ('calib', 'calmix', 'rawbase'):
        eps, gamma = params[:2]
        members = np.stack([m if raw else calibrate(m, data['prior'], eps, gamma)
                            for m, raw in zip(members, data['raw'])])
        if kind != 'rawbase':
            mean = members.mean(0)
    if kind == 'calib':
        probabilities, method = mean.astype(np.float32), 'bce'
    elif kind in ('mix', 'calmix', 'rawbase'):
        temperature, weight = params[-2:]
        count = len(members)
        probabilities = np.concatenate([mean[:, None], members.transpose(1, 0, 2)], 1)
        probabilities = probabilities.reshape(len(mean), -1).astype(np.float32)

        def method(queries, packed):
            parts = packed.reshape(len(packed), count + 1, 40)
            base = bce_distances(queries, parts[:, 0]).astype(np.float64)
            stacked = np.stack([bce_distances(queries, parts[:, 1 + m]).astype(np.float64) for m in range(count)])
            low = stacked.min(0)  # log-mean-exp, stabilized
            mixture = low - temperature * np.log(np.mean(np.exp(-(stacked - low) / temperature), axis=0))
            return (1 - weight) * base + weight * mixture
    elif kind in ('gtemp', 'gweight', 'sub', 'block'):
        members = np.stack([m if raw else calibrate(m, data['prior'], 0.05, 0.0)
                            for m, raw in zip(members, data['raw'])])
        mean = members.mean(0)
        count = len(members)
        probabilities = np.concatenate([mean[:, None], members.transpose(1, 0, 2)], 1)
        probabilities = probabilities.reshape(len(mean), -1).astype(np.float32)
        group = np.ones(40)

        def method(queries, packed, kind=kind, params=params):
            parts = packed.reshape(len(packed), count + 1, 40).astype(np.float64)
            averaged, each = parts[:, 0], [parts[:, 1 + m] for m in range(count)]
            mixture = soft_min(np.stack([weighted_bce(queries, p) for p in each]), 4.0)
            if kind == 'gtemp':
                t = group.copy(); t[COLOURS] = params[0]; t[HAIR_BAG_GLASSES] = params[1]
                base = weighted_bce(queries, 1 / (1 + np.exp(-logit(averaged) / t)))
            elif kind == 'gweight':
                w = group.copy(); w[COLOURS] = params[0]; w[HAIR_BAG_GLASSES] = params[1]
                base = weighted_bce(queries, averaged, w)
            else:
                base = weighted_bce(queries, averaged)
                if kind == 'sub':
                    subsets = [np.mean([each[j] for j in range(count) if j != m], axis=0) for m in range(count)]
                    extra = soft_min(np.stack([weighted_bce(queries, p) for p in subsets]), params[0])
                else:
                    extra = 0.0
                    for category in CATEGORIES:
                        columns = list(category)
                        temperature = 4.0 if params[0] == 0 else 4.0 * len(columns) / 40
                        extra = extra + soft_min(np.stack([weighted_bce(queries, p, columns=columns)
                                                           for p in each]), temperature)
                mixture = (1 - params[1]) * mixture + params[1] * extra
            return 0.5 * base + 0.5 * mixture
    elif kind == 'irratext':
        members = np.stack([m if raw else calibrate(m, data['prior'], 0.05, 0.0)
                            for m, raw in zip(members, data['raw'])])
        count = len(members)
        probabilities = np.concatenate([members.mean(0)[:, None], members.transpose(1, 0, 2)], 1)
        probabilities = probabilities.reshape(len(members[0]), -1).astype(np.float32)

        def method(queries, packed, lam=params[0]):
            return calmix_distance(queries, packed, count) - lam * text_similarity(queries, data)
    elif kind == 'nearmiss':
        members = np.stack([m if raw else calibrate(m, data['prior'], 0.05, 0.0)
                            for m, raw in zip(members, data['raw'])])
        count = len(members)
        probabilities = np.concatenate([members.mean(0)[:, None], members.transpose(1, 0, 2)], 1)
        probabilities = probabilities.reshape(len(members[0]), -1).astype(np.float32)
        labels = binary_matrix(data['labels'], 'labels').astype(np.float64)

        def method(queries, packed, mode=int(params[0]), a=params[1], b=params[2]):
            d = calmix_distance(queries, packed, count)
            if 'text' in data:
                d = d - 5.0 * text_similarity(queries, data)
            q = binary_matrix(queries, 'queries').astype(np.float64)
            if mode == 0:  # oracle: exact matches keep their slots, the others refill theirs by true agreement
                agreement = 1 - np.abs(q[:, None, :] - labels[None]).mean(2)
                out = np.empty_like(d)
                for row in range(len(d)):
                    order = np.argsort(d[row], kind='stable')
                    exact = agreement[row, order] == 1
                    others = order[~exact]
                    others = others[np.argsort(-agreement[row, others], kind='stable')]
                    filled = order.copy()
                    filled[~exact] = others
                    out[row, filled] = np.arange(len(filled))
                return out
            p = np.clip(packed.reshape(len(packed), count + 1, 40)[:, 0].astype(np.float64), EPS, 1 - EPS)
            r_on, r_off = p, 1 - p  # probability of the query's value when the query bit is 1 / 0
            if mode == 1:  # A: Hamming order inside distance bins of width a
                hamming = (1 - r_off).sum(1)[None, :] + q @ (r_off - r_on).T
                bins = np.floor(d / a)
                return bins * 1000 + hamming * 10 + (d / a - bins)
            # B: D - min(b, 0.5 * log(1 + a * sum_a (1 - r_a) / r_a))
            odds = (r_on / r_off)[None] * (1 - q[:, None, :]) + (r_off / r_on)[None] * q[:, None, :]
            return d - np.minimum(b, 0.5 * np.log1p(a * odds.sum(2)))
    elif kind == 'textfuse':
        members = np.stack([m if raw else calibrate(m, data['prior'], 0.05, 0.0)
                            for m, raw in zip(members, data['raw'])])
        count = len(members)
        probabilities = np.concatenate([members.mean(0)[:, None], members.transpose(1, 0, 2)], 1)
        probabilities = probabilities.reshape(len(members[0]), -1).astype(np.float32)

        def method(queries, packed, mode=int(params[0]), v=params[1]):
            d = calmix_distance(queries, packed, count)
            c = text_similarity(queries, data)
            if mode == 0:
                return (1 - v) * zscore(d) - v * zscore(c)
            if mode == 1:
                return d - v * c / c.std(1, keepdims=True).clip(min=1e-9)
            return (1 - v) * ranks(d) + v * ranks(-c)
    elif kind == 'cap':
        probabilities = mean.astype(np.float32)

        def method(queries, p):
            return capped_distances(queries, p, params[0])
    else:
        probabilities, method = mean.astype(np.float32), 'bce'
    with contextlib.redirect_stdout(io.StringIO()):
        report = evaluate_split(data['queries'], data['labels'], data['eval_ids'], probabilities,
                                data['image_names'].tolist(), by_domain=False, retrieval=method)
    return variant, domain, report['metrics']


def floats(text):
    return [float(x) for x in text.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('members', nargs='+')
    parser.add_argument('--domains', nargs='+', default=['Market1501', 'PETA'], choices=DOMAINS)
    parser.add_argument('--raw', nargs='*', default=[], help='Member letters (A, B, ...) left uncalibrated')
    parser.add_argument('--eps', type=floats, default=[0.0, 0.05, 0.1])
    parser.add_argument('--gamma', type=floats, default=[0.0, 0.5, 1.0])
    parser.add_argument('--temperature', type=floats, default=[1.0, 2.0, 4.0])
    parser.add_argument('--mix-weight', type=floats, default=[0.25, 0.5, 1.0])
    parser.add_argument('--cap', type=floats, default=[2.0, 3.0, 4.0])
    parser.add_argument('--combo', nargs='*', type=floats, default=[],
                        help='eps,gamma,T,w settings scored as calib followed by mix')
    for name, size, text in (('gtemp', 2, 'tc,th'), ('gweight', 2, 'wc,wh'), ('sub', 2, 'Ts,eta'),
                             ('block', 2, 'mode,eta')):
        parser.add_argument(f'--{name}', nargs='*', type=floats, default=[], help=f'{text} settings (calmix base)')
    parser.add_argument('--irra-text-cache', help='.npz pattern with {domain} from irra_lodo.py export --globals')
    parser.add_argument('--irratext', type=floats, default=[], help='Comma-separated lambda values for the image-text score')
    parser.add_argument('--nearmiss', nargs='*', type=floats, default=[],
                        help='mode,a,b on calmix + text 5 (see above): 0,0,0 oracle bound; 1,delta,0 bins; 2,eta,delta')
    parser.add_argument('--textfuse', nargs='*', type=floats, default=[],
                        help='mode,value settings of other image-text fusions (see above)')
    parser.add_argument('--rawbase', nargs='*', type=floats, default=[],
                        help='eps,gamma,T,w settings with the BCE term on uncalibrated probabilities')
    parser.add_argument('--min-positives', type=int, default=1,
                        help='Score only queries with at least this many gallery images (test-like)')
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    letters = [chr(ord('A') + i) for i in range(len(args.members))]
    if any(r not in letters for r in args.raw):
        parser.error(f'--raw takes member letters among {letters}')
    train_names, train_labels, _, train_ids = load_split(args.data_dir, 'train')
    variants = [('base', ())]
    variants += [('calib', (e, g)) for e in args.eps for g in args.gamma if (e, g) != (0.0, 0.0)]
    variants += [('mix', (t, w)) for t in args.temperature for w in args.mix_weight]
    variants += [('cap', (c,)) for c in args.cap]
    if any(len(c) != 4 for c in args.combo):
        parser.error('--combo takes eps,gamma,T,w')
    variants += [('calmix', tuple(c)) for c in args.combo]
    if any(len(c) != 4 for c in args.rawbase):
        parser.error('--rawbase takes eps,gamma,T,w')
    variants += [('rawbase', tuple(c)) for c in args.rawbase]
    if args.irratext and not args.irra_text_cache:
        parser.error('--irratext needs --irra-text-cache')
    variants += [('irratext', (lam,)) for lam in args.irratext]
    if any(len(c) != 2 or c[0] not in (0, 1, 2) for c in args.textfuse):
        parser.error('--textfuse takes mode,value with mode 0, 1 or 2')
    if args.textfuse and not args.irra_text_cache:
        parser.error('--textfuse needs --irra-text-cache')
    variants += [('textfuse', tuple(c)) for c in args.textfuse]
    if any(len(c) != 3 or c[0] not in (0, 1, 2) for c in args.nearmiss):
        parser.error('--nearmiss takes mode,a,b with mode 0 (oracle), 1 (bins) or 2 (one-error term)')
    variants += [('nearmiss', tuple(c)) for c in args.nearmiss]
    for name in ('gtemp', 'gweight', 'sub', 'block'):
        settings = getattr(args, name)
        if any(len(c) != 2 for c in settings):
            parser.error(f'--{name} takes two numbers per setting')
        variants += [(name, tuple(c)) for c in settings]
    tasks = []
    for domain in args.domains:
        caches = [load_predictions(cache_path(m, domain)) for m in args.members]
        if any(not np.array_equal(c['image_names'], caches[0]['image_names']) for c in caches):
            parser.error(f'{domain}: gallery order differs between members')
        _, seen, _ = select_domains(train_names, train_labels, train_ids, [d for d in DOMAINS if d != domain])
        data = {key: caches[0][key] for key in ('queries', 'labels', 'ids', 'image_names')}
        data['members'] = np.stack([c['probs'] for c in caches]).astype(np.float64)
        data['prior'] = seen.mean(0).astype(np.float64)
        data['raw'] = [letter in args.raw for letter in letters]
        # Only queries with at least --min-positives gallery images are scored (relevance comes from
        # the labels, so other images stay as distractors): their ids are pointed at a kept query,
        # which only drops them from the list.
        counts = np.bincount(data['ids'], minlength=len(data['queries']))
        keep = counts >= args.min_positives
        data['eval_ids'] = np.where(keep[data['ids']], data['ids'], np.flatnonzero(keep)[0])
        if args.min_positives > 1:
            print(f'{domain}: {keep.sum()} of {(counts > 0).sum()} queries with >= {args.min_positives} images', flush=True)
        if args.irra_text_cache:
            text_cache = np.load(cache_path(args.irra_text_cache, domain))
            if not np.array_equal(text_cache['image_names'], caches[0]['image_names']):
                parser.error(f'{domain}: IRRA embedding gallery order differs')
            data['image'] = text_cache['image'].astype(np.float64)
            data['text'] = text_cache['text'].astype(np.float64)
            data['text_row'] = {tuple(int(x) for x in q): i for i, q in enumerate(text_cache['queries'])}
            # Scale hint: the teammate's 0.24 / 0.76 z-score fusion as a fixed weight on cosine,
            # lam0 = (0.24 / 0.76) * mean over queries of std(calmix distance) / std(cosine).
            members = np.stack([m if raw else calibrate(m, data['prior'], 0.05, 0.0)
                                for m, raw in zip(data['members'], data['raw'])])
            packed = np.concatenate([members.mean(0)[:, None], members.transpose(1, 0, 2)], 1)
            packed = packed.reshape(len(packed), -1)
            sample = data['queries'][np.unique(data['ids'])][:200]
            ratio = np.mean(calmix_distance(sample, packed, len(members)).std(1)
                            / text_similarity(sample, data).std(1))
            print(f'{domain}: lambda0 (teammate fusion scale) = {0.24 / 0.76 * ratio:.3f}', flush=True)
        tasks += [(v, domain, data) for v in variants]
    results = {}
    with ProcessPoolExecutor(args.jobs) as executor:
        for variant, domain, metrics in executor.map(score, tasks):
            results[variant, domain] = metrics
    for letter, member in zip(letters, args.members):
        print(f'{letter} = {member}' + ('  (raw)' if letter in args.raw else ''))
    print('\nheld-out mADM per split, then means over splits')
    print(f'{"variant":22s} ' + ' '.join(f'{d:>10s}' for d in args.domains)
          + '       mADM     mAP  ADM-AP  Rank-1    mINP')
    for variant in variants:
        kind, params = variant
        name = kind if not params else f'{kind}({",".join(f"{p:g}" for p in params)})'
        per = [results[variant, d] for d in args.domains]
        means = [np.mean([m[k] for m in per]) for k in ('mADM', 'mAP', 'Rank-1', 'mINP')]
        means.insert(2, means[0] - means[1])  # near-miss share of mADM (mADM - mAP)
        print(f'{name:22s} ' + ' '.join(f'{m["mADM"]:10.4f}' for m in per)
              + '   ' + ' '.join(f'{v:7.4f}' for v in means))


if __name__ == '__main__':
    main()
