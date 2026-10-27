"""SPEC.md retrieval metrics with exact vectors and stable gallery-order ties."""

import numpy as np
import torch
import torch.nn.functional as F



def metrics_for_ranking(ranking, query, gallery_attributes, ranks=(1, 5, 10)):
    gallery = torch.as_tensor(gallery_attributes, device='cpu')
    query = torch.as_tensor(query, device='cpu')
    ranking = torch.as_tensor(ranking, device='cpu')
    if gallery.ndim != 2 or gallery.shape[1] != 40 or query.shape != (40,):
        raise ValueError('Metrics require query/gallery vectors with 40 binary attributes')
    if ranking.shape != (len(gallery),):
        raise ValueError('Ranking must cover the whole gallery')
    if ranking.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('Ranking must contain integer gallery indices')
    ranking = ranking.long()
    if (ranking < 0).any() or (ranking >= len(gallery)).any():
        raise ValueError('Ranking indices are outside the gallery')
    if not torch.bincount(ranking, minlength=len(gallery)).eq(1).all():
        raise ValueError('Ranking must be a permutation of all gallery indices')
    if not ((gallery == 0) | (gallery == 1)).all() or not ((query == 0) | (query == 1)).all():
        raise ValueError('Metric annotations must be binary')
    agreements = gallery.eq(query).sum(dim=1)
    matches = agreements.eq(40)[ranking]
    relevant_positions = matches.nonzero(as_tuple=True)[0]
    relevant_count = len(relevant_positions)
    if not relevant_count:
        raise ValueError('Every UPAR query must have at least one exact gallery match')
    dom = agreements.double() / 40.0
    mean_dom, maximum_dom = dom.mean(), dom.max()
    denominator = maximum_dom - mean_dom
    # With an exact match and constant dom, the whole gallery is a perfect match.
    ndom = (torch.ones_like(dom) if denominator == 0 else
            ((dom - mean_dom) / denominator).clamp_min(0.0))
    positions = torch.arange(1, len(gallery) + 1, dtype=torch.float64)
    precision = matches.double().cumsum(0) / positions
    match_degree = ndom[ranking].cumsum(0) / positions
    first_rank = int(relevant_positions[0]) + 1
    last_rank = int(relevant_positions[-1]) + 1
    result = {'mADM': float(match_degree[relevant_positions].mean()),
              'mAP': float(precision[relevant_positions].mean()),
              'mINP': relevant_count / last_rank}
    result.update({f'Rank-{rank}': float(first_rank <= rank) for rank in ranks})
    return result


def retrieval_metrics(query_features, gallery_features, query_attributes,
                      gallery_attributes, query_batch_size=32):
    """Evaluate in query chunks instead of allocating Q x G x 40 labels."""
    if not len(query_features) or not len(gallery_features):
        raise ValueError('Retrieval evaluation requires nonempty query and gallery sets')
    if len(query_features) != len(query_attributes) or len(gallery_features) != len(gallery_attributes):
        raise ValueError('Feature and annotation row counts differ')
    queries = F.normalize(query_features.float().cpu(), dim=1)
    gallery = F.normalize(gallery_features.float().cpu(), dim=1)
    totals = {name: 0.0 for name in ('mADM', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'mINP')}
    for start in range(0, len(queries), query_batch_size):
        similarities = queries[start:start + query_batch_size] @ gallery.t()
        if not torch.isfinite(similarities).all():
            raise ValueError('Non-finite retrieval similarities')
        # CPU mergesort keeps gallery order for equal scores, including older torch
        # environments that do not expose the stable argsort keyword.
        rankings = torch.from_numpy(np.argsort(-similarities.numpy(), axis=1, kind='mergesort'))
        for offset, ranking in enumerate(rankings):
            metrics = metrics_for_ranking(ranking, query_attributes[start + offset], gallery_attributes)
            for name, value in metrics.items():
                totals[name] += value
    return {name: value / len(queries) for name, value in totals.items()}


