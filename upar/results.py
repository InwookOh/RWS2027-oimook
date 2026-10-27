"""Public-validation prediction caches and experiment result tables."""
import csv
import json
import os
from pathlib import Path

import numpy as np

from .attributes import ATTRIBUTES


def save_predictions(path, probs, labels, queries, ids, names, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, probs=np.asarray(probs, dtype=np.float32),
                            labels=np.asarray(labels, dtype=np.uint8), queries=np.asarray(queries, dtype=np.uint8),
                            ids=np.asarray(ids, dtype=np.int64), image_names=np.asarray(names),
                            domain=np.asarray([n.split('/')[0] for n in names]),
                            attribute_names=np.asarray(ATTRIBUTES), metadata=np.asarray(json.dumps(metadata)))
    os.replace(temporary, path)


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    p, y, q, ids, names = (data[k] for k in ('probs', 'labels', 'queries', 'ids', 'image_names'))
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 40 or len(names) != len(p):
        raise ValueError('Invalid prediction cache shape')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid prediction probabilities')
    if not np.isin(y, (0, 1)).all() or not np.isin(q, (0, 1)).all():
        raise ValueError('Cache labels and queries must be binary')
    if ids.shape != (len(p),) or np.any(ids < 0) or np.any(ids >= len(q)) or not np.array_equal(y, q[ids]):
        raise ValueError('Cache query mapping is invalid')
    if list(data['attribute_names']) != list(ATTRIBUTES):
        raise ValueError('Cache attribute order mismatch')
    if not np.array_equal(data['domain'], np.asarray([n.split('/')[0] for n in names])):
        raise ValueError('Cache domain mapping mismatch')
    data['metadata'] = json.loads(str(data['metadata']))
    return data


def append_results(path, report, context):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['run', 'epoch', 'seed', 'amp', 'retrieval', 'subset_run', 'cache', 'gallery',
              'images', 'queries', 'mADM', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'mINP',
              'training_seconds', 'gallery_inference_seconds', 'metrics_seconds']
    groups = {'all': report, **report.get('by_domain', {})}
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        for domain, result in groups.items():
            row = {**context, 'gallery': domain, 'images': result['images'],
                   'queries': result['queries'], **result['metrics']}
            writer.writerow({key: row.get(key, '') for key in fields})
