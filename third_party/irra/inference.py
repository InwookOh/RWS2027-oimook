#!/usr/bin/env python3
"""Run Epoch 2 FrozenMultiScaleLocalAttr inference on UPAR Track 2 validation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from upar.attributes import attribute_vector
from upar.data import GalleryDataset, UPARSplit
from upar.frozen_multiscale import FrozenLocalConfig, FrozenMultiScaleLocalModel
from upar.local_scoring import zscore
from upar.metrics import metrics_for_ranking
from upar.prompts import encode_query
from upar.tokenizer import PromptTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "experiments/frozen_localattr_e5_20261013_041809_KST/epoch_002.pth"
LOCAL_NLL_WEIGHT = 0.76
GLOBAL_WEIGHT = 0.24
TEXT_BATCH_SIZE = 64
QUERY_SCORE_CHUNK = 16


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return result


def _read_checkpoint(path):
    try:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as error:
        # weights_only was introduced after the project's original PyTorch version.
        if "weights_only" not in str(error):
            raise
        state = torch.load(str(path), map_location="cpu")
    if not isinstance(state, dict) or len(state) != 532:
        raise ValueError("Expected the complete 532-tensor Epoch 2 model state_dict")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise TypeError("Checkpoint must be a plain model state_dict")
    if any(value.is_floating_point() and value.dtype != torch.float32
           for value in state.values()):
        raise TypeError("Checkpoint floating-point tensors must be FP32")
    return state


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Checkpoint not found: %s" % checkpoint)

    cuda_backend = getattr(torch.backends, "cuda", None)
    if cuda_backend is not None and hasattr(cuda_backend, "matmul"):
        cuda_backend.matmul.allow_tf32 = False
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = False

    config = FrozenLocalConfig(
        attr_loss_weight=0.5,
        local_heads=8,
        fusion_weight=LOCAL_NLL_WEIGHT,
        attribute_score="nll",
        multiscale_blocks=(8, 16, 24),
        initial_scale_weights=(0.2, 0.2, 0.6),
    )
    model = FrozenMultiScaleLocalModel(config=config).float()
    state = _read_checkpoint(checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device)
    return model


def _encode_queries(model, queries, tokenizer, device):
    # Match the Codabench adapter: encode unique binary queries, then restore rows.
    query_rows = np.asarray([attribute_vector(row) for row in queries], dtype=np.int64)
    if not len(query_rows):
        raise ValueError("Validation split contains no queries")
    unique_attributes, inverse = np.unique(query_rows, axis=0, return_inverse=True)
    prompt_rows = [encode_query(row, tokenizer).token_ids for row in unique_attributes]
    token_ids = torch.as_tensor(prompt_rows, dtype=torch.long)
    feature_parts = []
    with torch.inference_mode():
        for start in range(0, len(token_ids), TEXT_BATCH_SIZE):
            features = model.encode_text(token_ids[start:start + TEXT_BATCH_SIZE].to(device))
            feature_parts.append(features.float().cpu())
    unique_features = F.normalize(torch.cat(feature_parts).float(), dim=1)
    return query_rows, unique_attributes, inverse, unique_features


def run_inference(checkpoint, data_root, output_dir, device="auto", batch_size=4, model=None):
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    device = _device(device)
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / name for name in
                    ("distances.npy", "metrics.json", "matrix_order.json")]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite inference outputs: %s" % ", ".join(existing))

    split = UPARSplit.read(data_root, "val")
    if model is None:
        model = load_model(checkpoint, device)
    else:
        model.eval().to(device)
    tokenizer = PromptTokenizer()
    query_attributes, unique_attributes, inverse, query_features = _encode_queries(
        model, split.queries, tokenizer, device)

    gallery_loader = DataLoader(
        GalleryDataset(split), batch_size=batch_size, shuffle=False, num_workers=0)
    gallery_feature_parts, gallery_logit_parts, gallery_attribute_parts = [], [], []
    print("Encoding gallery: %d images on %s" % (len(split.images), device), flush=True)
    with torch.inference_mode():
        for batch_index, (attributes, images) in enumerate(gallery_loader, start=1):
            features, logits = model.encode_image_and_attributes(
                images.to(device=device, dtype=torch.float32))
            features, logits = features.float().cpu(), logits.float().cpu()
            if not torch.isfinite(features).all().item() or not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite image feature or attribute logit at gallery batch %d"
                                 % batch_index)
            gallery_feature_parts.append(features)
            gallery_logit_parts.append(logits)
            gallery_attribute_parts.append(attributes)
            completed = min(batch_index * batch_size, len(split.images))
            if completed == len(split.images) or completed // 500 != (completed - batch_size) // 500:
                print("Gallery %d/%d" % (completed, len(split.images)), flush=True)

    gallery_features = F.normalize(torch.cat(gallery_feature_parts).float(), dim=1)
    gallery_logits = torch.cat(gallery_logit_parts).float()
    gallery_attributes = torch.cat(gallery_attribute_parts).numpy()
    expected_gallery_attributes = np.asarray([row.attributes for row in split.images], dtype=np.int64)
    if (gallery_features.shape[0] != len(split.images)
            or gallery_logits.shape != (len(split.images), 40)
            or not np.array_equal(gallery_attributes, expected_gallery_attributes)):
        raise RuntimeError("Gallery output order or shape differs from gt.csv")

    distances = np.lib.format.open_memmap(
        str(output_paths[0]), mode="w+", dtype=np.float64,
        shape=(len(query_attributes), len(split.images)))
    logits64 = gallery_logits.double()
    negative_log_prob = F.logsigmoid(-logits64).sum(dim=1)
    metric_names = ("mADM", "mAP", "Rank-1", "Rank-5", "Rank-10")
    totals = {name: 0.0 for name in metric_names}
    query_attributes64 = torch.from_numpy(query_attributes).double()

    print("Fusing scores: local NLL weight=%.2f, global weight=%.2f" %
          (LOCAL_NLL_WEIGHT, GLOBAL_WEIGHT), flush=True)
    for start in range(0, len(unique_attributes), QUERY_SCORE_CHUNK):
        end = min(start + QUERY_SCORE_CHUNK, len(unique_attributes))
        global_scores = (query_features[start:end] @ gallery_features.t()).numpy()
        local_nll = (negative_log_prob.unsqueeze(0) +
                     torch.from_numpy(unique_attributes[start:end]).double() @ logits64.t()).numpy()
        if not np.isfinite(global_scores).all() or not np.isfinite(local_nll).all():
            raise ValueError("Nonfinite global or local retrieval score")
        fused_similarity = (GLOBAL_WEIGHT * zscore(global_scores) +
                            LOCAL_NLL_WEIGHT * zscore(local_nll))
        for offset, unique_index in enumerate(range(start, end)):
            rows = np.flatnonzero(inverse == unique_index)
            distances[rows] = -fused_similarity[offset]
        del global_scores, local_nll, fused_similarity

    distances.flush()
    for start in range(0, len(query_attributes), QUERY_SCORE_CHUNK):
        if not np.isfinite(distances[start:start + QUERY_SCORE_CHUNK]).all():
            raise ValueError("Nonfinite final distance matrix")

    # Ascending distance uses the same stable gallery-order tie break as the adapter.
    for query_index, row in enumerate(distances):
        ranking = np.argsort(row, kind="mergesort")
        result = metrics_for_ranking(
            ranking, query_attributes64[query_index], gallery_attributes)
        for name in metric_names:
            totals[name] += result[name]
    metrics = {name: value / len(query_attributes) for name, value in totals.items()}
    report = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "data_root": str(data_root),
        "device": str(device),
        "queries": len(query_attributes),
        "gallery": len(split.images),
        "fusion": {"global_weight": GLOBAL_WEIGHT, "local_nll_weight": LOCAL_NLL_WEIGHT},
        "matrix": {
            "file": output_paths[0].name,
            "shape": [len(query_attributes), len(split.images)],
            "dtype": "float64",
            "values": "negative fused similarity; lower ranks first",
        },
        "metrics": metrics,
    }
    order = {
        "query_order": "data rows in annotations/task2/val/queries.csv",
        "query_attributes": [list(row) for row in query_attributes.tolist()],
        "gallery_order": "data rows in annotations/task2/val/gt.csv",
        "gallery_image_paths": [record.relative_path for record in split.images],
    }
    output_paths[1].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    output_paths[2].write_text(json.dumps(order, indent=2) + "\n", encoding="utf-8")
    del distances
    print(json.dumps(report, indent=2), flush=True)
    print("Saved results to %s" % output_dir, flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="full Epoch 2 state_dict (default: %(default)s)")
    parser.add_argument("--data-root", required=True,
                        help="UPAR root containing annotations/task2/val and gallery images")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, or cuda:N (default: auto)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="gallery image inference batch size (default: %(default)s)")
    parser.add_argument("--output-dir", default="outputs/frozen_localattr_epoch002",
                        help="new or existing directory for inference outputs")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_inference(args.checkpoint, args.data_root, args.output_dir,
                  device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Inference failed: %s" % error, file=sys.stderr, flush=True)
        raise
