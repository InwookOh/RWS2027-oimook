"""Unchanged QueryBalanced normalized-global retrieval, with FP32 GPU chunking."""
import torch
from .metrics import retrieval_metrics
from .precision import assert_model_fp32


@torch.inference_mode()
def evaluate(model, gallery_loader, query_loader, query_batch_size=32, micro_batch_size=4, progress=None):
    if micro_batch_size < 1: raise ValueError('Positive inference micro batch required')
    training = model.training
    model.eval()
    assert_model_fp32(model)
    device = next(model.parameters()).device
    collections = []
    try:
        for label, loader, encode in (('query', query_loader, model.encode_text),
                                      ('gallery', gallery_loader, model.encode_image)):
            features, annotations = [], []
            observed = 0
            for attrs, values in loader:
                for start in range(0, len(values), micro_batch_size):
                    encoded = encode(values[start:start + micro_batch_size].to(device))
                    if encoded.dtype != torch.float32 or not torch.isfinite(encoded).all():
                        raise FloatingPointError('Non-finite/non-FP32 retrieval feature')
                    features.append(encoded.cpu())
                annotations.append(attrs.cpu())
                observed += len(values)
                if progress is not None: progress(label, observed, len(loader.dataset))
            if not features: raise ValueError('Empty validation collection')
            collections.append((torch.cat(features), torch.cat(annotations)))
        (query, qa), (gallery, ga) = collections
        if query.ndim != 2 or gallery.ndim != 2 or query.shape[1] != gallery.shape[1]:
            raise ValueError('Image/text retrieval dimensions differ')
        return retrieval_metrics(query, gallery, qa, ga, query_batch_size)
    finally:
        model.train(training)
