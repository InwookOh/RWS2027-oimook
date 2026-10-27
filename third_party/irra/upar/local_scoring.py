"""Stable attribute scores and per-query fusion without Q x G x 40 tensors."""
import math
import numpy as np
import torch
import torch.nn.functional as F


def attribute_scores(queries, logits):
    q, z = torch.as_tensor(queries).double().cpu(), torch.as_tensor(logits).double().cpu()
    if q.ndim != 2 or q.shape[1] != 40 or z.ndim != 2 or z.shape[1] != 40:
        raise ValueError('Expected query [Q,40], gallery logits [G,40]')
    if not ((q == 0) | (q == 1)).all() or not torch.isfinite(z).all():
        raise ValueError('Invalid query attributes or nonfinite logits')
    log_negative = F.logsigmoid(-z)
    # log(sigmoid(z))-log(sigmoid(-z)) = z, stably, including extreme logits.
    nll_score = log_negative.sum(1)[None, :] + q @ z.t()
    p = torch.sigmoid(z)
    agreement = ((1-p).sum(1)[None, :] + q @ (2*p-1).t()) / 40
    if not torch.isfinite(nll_score).all() or not torch.isfinite(agreement).all():
        raise ValueError('Nonfinite attribute score')
    return nll_score.numpy(), agreement.numpy()


def zscore(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite [Q,G]')
    std = scores.std(axis=1, keepdims=True, ddof=0)
    return (scores-scores.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)


def fuse(global_scores, attribute_score, weight):
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Fusion weight must be in [0,1]')
    if np.shape(global_scores) != np.shape(attribute_score):
        raise ValueError('Global/attribute score shape mismatch')
    if not np.isfinite(global_scores).all() or not np.isfinite(attribute_score).all():
        raise ValueError('Nonfinite fusion inputs')
    # Exact endpoints preserve the original ranking and stable tie semantics.
    if weight == 0:
        return np.asarray(global_scores)
    if weight == 1:
        return np.asarray(attribute_score)
    return (1-weight)*zscore(global_scores) + weight*zscore(attribute_score)


def stable_rank(scores):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite retrieval scores')
    return np.argsort(-scores, axis=-1, kind='mergesort')
