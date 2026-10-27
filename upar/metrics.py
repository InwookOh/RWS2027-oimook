"""Metrics from the supplied UPAR 2027 Track 2 Evaluation specification.

Ranks are one-based and distance ties keep gallery order. ADM normalizes the
agreement over the full gallery and samples prefix means at exact-match ranks.
"""
import numpy as np

from .retrieval import binary_matrix, l1_distances, query_distances


def query_metrics(query, gallery_labels, distances):
    labels = binary_matrix(gallery_labels, 'gallery labels')
    query = binary_matrix(np.asarray(query).reshape(1, -1), 'query')[0]
    distances = np.asarray(distances)
    if distances.shape != (len(labels),) or not np.isfinite(distances).all():
        raise ValueError('Expected one finite distance per gallery image')
    relevant = np.all(labels == query, axis=1)
    count = int(relevant.sum())
    if not count:
        raise ValueError('Every evaluated query must have at least one exact match')
    dom = np.mean(labels == query, axis=1, dtype=np.float64)
    return _metrics_from_agreement(dom, distances)


def _metrics_from_agreement(dom, distances):
    relevant = dom == 1
    count = int(relevant.sum())
    if not count:
        raise ValueError('Every evaluated query must have at least one exact match')
    order = np.argsort(distances, kind='stable')
    positions = np.flatnonzero(relevant[order])
    ranks = positions.astype(np.float64) + 1
    mean, maximum = dom.mean(), dom.max()
    if maximum == mean:
        # A valid query in a constant-agreement gallery implies all are exact matches.
        ndom = np.ones(len(dom), dtype=np.float64)
    else:
        ndom = np.maximum(0, (dom - mean) / (maximum - mean))
    prefix = np.cumsum(ndom[order], dtype=np.float64)
    return {
        'mADM': float(np.mean(prefix[positions] / ranks)),
        'mAP': float(np.mean(np.arange(1, count + 1) / ranks)),
        'Rank-1': float(ranks[0] <= 1),
        'Rank-5': float(ranks[0] <= 5),
        'Rank-10': float(ranks[0] <= 10),
        'mINP': float(count / ranks[-1]),
    }


def evaluate_retrieval(queries, labels, probabilities, chunk_size=64, retrieval='l1'):
    queries = binary_matrix(queries, 'queries')
    labels = binary_matrix(labels, 'gallery labels')
    if len(labels) != len(probabilities) or not len(labels) or not len(queries):
        raise ValueError('Nonempty aligned queries/gallery are required')
    if chunk_size < 1:
        raise ValueError('Query chunk size must be positive')
    totals = {k: 0.0 for k in ('mADM', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'mINP')}
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start:start + chunk_size]
        # `retrieval` is a method name or a callable(queries, probabilities) -> Q x G distances.
        block = retrieval(chunk, probabilities) if callable(retrieval) else query_distances(chunk, probabilities, retrieval)
        agreement = 1 - l1_distances(chunk, labels).astype(np.float64) / 40
        for dom, distance in zip(agreement, block):
            for key, value in _metrics_from_agreement(dom, distance).items():
                totals[key] += value
        if start == 0 or start + chunk_size >= len(queries) or start % 512 == 0:
            print(f'Retrieval evaluation: {min(start + chunk_size, len(queries))}/{len(queries)}', flush=True)
    return {key: value / len(queries) for key, value in totals.items()}
