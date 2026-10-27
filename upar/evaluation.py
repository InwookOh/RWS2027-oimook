from collections import Counter

import numpy as np

from .metrics import evaluate_retrieval


def evaluate_split(queries, labels, ids, probabilities, image_names, chunk_size=64, by_domain=True, retrieval='l1'):
    present = np.unique(ids)
    result = {
        'metrics': evaluate_retrieval(queries[present], labels, probabilities, chunk_size, retrieval),
        'images': len(labels), 'queries': len(present),
        'queries_without_matches': len(queries) - len(present),
        'images_by_domain': dict(Counter(name.split('/')[0] for name in image_names)),
    }
    if by_domain:
        result['by_domain'] = {}
        domains = np.asarray([name.split('/')[0] for name in image_names])
        for domain in sorted(set(domains)):
            mask = domains == domain
            domain_queries = np.unique(ids[mask])
            result['by_domain'][domain] = {
                'metrics': evaluate_retrieval(queries[domain_queries], labels[mask], probabilities[mask], chunk_size, retrieval),
                'images': int(mask.sum()), 'queries': len(domain_queries),
            }
    return result
