#!/usr/bin/env python3
"""Strictly check the Epoch-2 start model and the frozen local-head update."""
import argparse
import gc
import hashlib
import json
import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.clip_model import CLIP
from upar.checkpoint import load_state_dict
from upar.config import ModelConfig
from upar.data import GalleryDataset, QueryDataset, TrainingDataset, UPARSplit
from upar.frozen_multiscale import FrozenLocalConfig, FrozenMultiScaleLocalModel, NEW_PREFIXES
from upar.local_scoring import attribute_scores, fuse
from upar.metrics import metrics_for_ranking
from upar.model import UPARModel
from upar.pretrained import CLIP_CONFIG
from upar.sampling import QueryBalancedSampler
from upar.tokenizer import PromptTokenizer
from upar.io import write_json


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def visible_gpu(expected_uuid, require_empty=True):
    result = subprocess.run(
        ['nvidia-smi', '--id=' + expected_uuid,
         '--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=30, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(',')]
        if values:
            rows.append({'name': values[0], 'uuid': values[1], 'memory_total_mib': int(values[2]),
                         'memory_used_mib': int(values[3]), 'utilization_gpu_percent': int(values[4])})
    if len(rows) != 1 or rows[0]['uuid'] != expected_uuid:
        raise RuntimeError('Expected the assigned GPU UUID %s, found %s' % (expected_uuid, rows))
    app_result = subprocess.run(
        ['nvidia-smi', '--id=' + expected_uuid,
         '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=30, check=True)
    applications = [line.strip() for line in app_result.stdout.splitlines() if line.strip()]
    limit = 512 if require_empty else 1024
    if rows[0]['memory_used_mib'] > limit or (require_empty and applications):
        raise RuntimeError('CUDA preflight stopped: visible GPU already has memory/apps: %s %s' %
                           (rows[0], applications))
    return rows[0]


def check_checkpoint_provenance(config):
    run_dir = Path(config['initial_run_dir'])
    path = Path(config['initial_checkpoint'])
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = file_sha256(path)
    if digest != config['initial_checkpoint_sha256']:
        raise ValueError('Epoch-2 checkpoint SHA256 differs from the configured immutable source')
    saved_config = load_json(run_dir / 'config.json')
    expected = saved_config['model']
    if tuple(expected['img_size']) != (378, 126) or expected['stride_size'] != 14:
        raise ValueError('Initial run config is not the requested ViT-L/14 geometry')
    lines = (run_dir / 'train.log').read_text(errors='replace').splitlines()
    validation_line = next((line for line in lines if 'Validation Epoch 2:' in line), None)
    if validation_line is None or str(path) not in validation_line:
        raise ValueError('Training log does not bind Validation Epoch 2 to this checkpoint')
    metric_lines = (run_dir / 'metrics.jsonl').read_text().splitlines()
    epoch_record = next((json.loads(line) for line in metric_lines
                         if json.loads(line).get('epoch') == 2), None)
    if epoch_record is None or epoch_record.get('weight') != str(path):
        raise ValueError('Metrics record does not bind completed Epoch 2 to this checkpoint')
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest,
            'epoch': 2, 'plain_state_dict_policy': saved_config.get('checkpoint_policy'),
            'validation_log': validation_line.strip(), 'metric_record': epoch_record,
            'model_config': expected}


def strict_load_new_model(state, seed):
    torch.manual_seed(seed)
    model = FrozenMultiScaleLocalModel(CLIP(**CLIP_CONFIG), CLIP_CONFIG, FrozenLocalConfig()).float()
    initialized = model.state_dict()
    old_keys, new_keys = set(state), set(initialized)
    absent = old_keys - new_keys
    added = new_keys - old_keys
    if absent:
        raise RuntimeError('New model is missing original QueryBalanced keys: %s' % sorted(absent)[:8])
    if not added or any(not key.startswith(NEW_PREFIXES) for key in added):
        raise RuntimeError('Unexpected keys added beyond multiscale/local branch: %s' % sorted(added)[:8])
    for name, tensor in state.items():
        if initialized[name].shape != tensor.shape or initialized[name].dtype != tensor.dtype:
            raise RuntimeError('Original parameter mismatch: %s %s != %s' %
                               (name, tuple(tensor.shape), tuple(initialized[name].shape)))
    expanded = dict(initialized)
    expanded.update(state)
    model.load_state_dict(expanded, strict=True)
    model.configure_frozen_training()
    frozen_bad = [name for name, parameter in model.named_parameters()
                  if not name.startswith(NEW_PREFIXES) and parameter.requires_grad]
    trainable_bad = [name for name, parameter in model.named_parameters()
                     if name.startswith(NEW_PREFIXES) and not parameter.requires_grad]
    if frozen_bad or trainable_bad:
        raise RuntimeError('Freeze invariant failed: frozen=%s trainable=%s' %
                           (frozen_bad[:8], trainable_bad[:8]))
    return model, sorted(added)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] /
                        'configs/frozen_localattr_e5.json')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=0,
                        help='Default is 32 on CUDA and 2 on CPU; GPU launch must use 32')
    args = parser.parse_args()
    config = load_json(args.config)
    if args.device == 'cuda':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = config['expected_gpu_uuid']
    gpu = visible_gpu(config['expected_gpu_uuid']) if args.device == 'cuda' else None
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no training may start')
    if args.device == 'cuda' and torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible GPU')
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(1)
    torch.manual_seed(config['train']['seed'])
    device = torch.device(args.device)
    started = time.time()
    report = {'started_unix': started, 'device': str(device), 'precision': 'FP32'}
    if gpu is not None:
        if torch.cuda.get_device_properties(0).name != gpu['name']:
            raise RuntimeError('NVML and PyTorch CUDA device names differ')
        report['gpu'] = gpu
        report['gpu_uuid'] = gpu['uuid']
    report['initial_checkpoint'] = check_checkpoint_provenance(config)
    state = load_state_dict(config['initial_checkpoint'])
    report['source_state_keys'] = len(state)
    if any(value.is_floating_point() and value.dtype != torch.float32 for value in state.values()):
        raise TypeError('Epoch 2 source must contain FP32 model tensors')

    # Required original-model strict load before any new head is attached.
    torch.manual_seed(config['train']['seed'])
    original = UPARModel(CLIP(**CLIP_CONFIG), config=ModelConfig(), with_map=True).float()
    original.load_state_dict(state, strict=True)
    original.eval()
    report['original_querybalanced_strict_load'] = True
    report['original_model_keys'] = len(original.state_dict())

    data_root = Path(config['data_root'])
    # A prior CPU preflight audits every path; here the sampled train/val images
    # are opened below, avoiding a long metadata-only walk while CUDA is reserved.
    train_split = UPARSplit.read(data_root, 'train', check_files=False)
    val_split = UPARSplit.read(data_root, 'val', check_files=False)
    tokenizer = PromptTokenizer()
    val_queries = QueryDataset(val_split, tokenizer)
    # Select two real validation queries and three real gallery images, including
    # at least one exact match for each query so official ranking metrics run.
    query_rows = [val_queries[i] for i in range(min(2, len(val_queries)))]
    selected_query_attrs = [tuple(row[0].tolist()) for row in query_rows]
    gallery_indices = []
    for attrs in selected_query_attrs:
        match = next((i for i, record in enumerate(val_split.images)
                      if tuple(record.attributes) == attrs), None)
        if match is None:
            raise RuntimeError('Validation query has no exact gallery match')
        if match not in gallery_indices:
            gallery_indices.append(match)
    for index in range(len(val_split.images)):
        if index not in gallery_indices:
            gallery_indices.append(index)
        if len(gallery_indices) == 3:
            break
    gallery_dataset = GalleryDataset(val_split)
    gallery_rows = [gallery_dataset[i] for i in gallery_indices]
    qattrs = torch.stack([row[0] for row in query_rows])
    qtokens = torch.stack([row[1] for row in query_rows])
    gattrs = torch.stack([row[0] for row in gallery_rows])
    images3 = torch.stack([row[1] for row in gallery_rows])
    with torch.inference_mode():
        baseline_text = original.encode_text(qtokens).float()
        baseline_image = original.base_model.encode_image(images3).float()
        baseline_global = torch.nn.functional.normalize(baseline_text, dim=1) @ \
            torch.nn.functional.normalize(baseline_image[:, 0], dim=1).t()
    report['real_validation_sample'] = {'queries': len(qattrs), 'gallery_images': len(images3),
                                        'gallery_indices': gallery_indices,
                                        'global_score_shape': list(baseline_global.shape)}

    # Free the duplicate base model before adding the new model on the selected device.
    del original
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model, added_keys = strict_load_new_model(state, config['train']['seed'])
    del state
    if device.type == 'cuda':
        # Dataset metadata and the CPU baseline are ready; recheck before the
        # sole full-model transfer to CUDA.
        visible_gpu(config['expected_gpu_uuid'], require_empty=False)
    model.to(device)
    if device.type == 'cuda':
        qtokens = qtokens.to(device)
        images3 = images3.to(device)
    report['expanded_model_strict_load'] = True
    report['new_state_keys'] = added_keys
    report['multiscale_shapes'] = {'joint_dim': 768}
    report['frozen_parameter_count'] = sum(p.numel() for n, p in model.named_parameters()
                                           if not n.startswith(NEW_PREFIXES))
    report['trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report['trainable_parameter_names'] = model.trainable_parameter_names()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise TypeError('The full model must remain FP32')

    model.eval()
    with torch.inference_mode():
        final3, intermediate3 = model.image_tokens_and_scales(images3)
        text2 = model.encode_text(qtokens).float()
        expanded_global = torch.nn.functional.normalize(text2, dim=1) @ \
            torch.nn.functional.normalize(final3[:, 0], dim=1).t()
        global_difference = float((expanded_global.cpu() - baseline_global.cpu()).abs().max())
        if not torch.allclose(expanded_global.cpu(), baseline_global.cpu(), atol=1e-6, rtol=1e-6):
            raise AssertionError('Global Epoch-2 retrieval scores changed after adding local modules')
        logits3 = model.local_from_tokens(final3, intermediate3)
    report['multiscale_shapes'] = {'block8': list(intermediate3[8].shape),
                                   'block16': list(intermediate3[16].shape),
                                   'block24_final': list(final3.shape),
                                   'joint_dim': int(final3.shape[-1])}
    if not torch.isfinite(logits3).all():
        raise FloatingPointError('Nonfinite Local Attribute logits')
    nll, agreement = attribute_scores(qattrs, logits3.cpu())
    global_np = expanded_global.cpu().numpy()
    fused = fuse(global_np, nll, 0.4)
    if fused.shape != (len(qattrs), len(images3)) or not np.isfinite(fused).all():
        raise RuntimeError('Global/local/fusion score output failed Q x G checks')
    sample_metrics = []
    for qi in range(len(qattrs)):
        ranking = np.argsort(-fused[qi], kind='mergesort')
        sample_metrics.append(metrics_for_ranking(ranking, qattrs[qi], gattrs))
    report['global_score_max_abs_difference'] = global_difference
    report['local_score_shape'] = list(nll.shape)
    report['agreement_score_shape'] = list(agreement.shape)
    report['fused_score_shape'] = list(fused.shape)
    report['score_dtype'] = str(fused.dtype)
    report['sample_fusion_metrics'] = sample_metrics
    report['old_qb_state_sha256_before_step'] = model.old_qb_state_digest()

    # The full effective Batch-32 preflight is mandatory on CUDA. CPU-only checks
    # use two images to keep a fallback diagnostic practical on this large encoder.
    batch_size = args.batch_size or (32 if device.type == 'cuda' else 2)
    if device.type == 'cuda' and batch_size != 32:
        raise ValueError('CUDA preflight must exercise Batch 32')
    sampler = QueryBalancedSampler(train_split, config['train']['seed'])
    sampler.set_epoch(1)
    train_dataset = TrainingDataset(train_split, tokenizer)
    indices = list(iter(sampler))[:batch_size]
    from torch.utils.data._utils.collate import default_collate
    batch = default_collate([train_dataset[index] for index in indices])
    images = batch['images'].to(device)
    attributes = batch['attributes'].to(device)
    names = model.trainable_parameter_names()
    optimizer = torch.optim.Adam([dict(params=[p for p in model.parameters() if p.requires_grad])],
                                 lr=config['train']['lr'], betas=tuple(config['train']['adam_betas']),
                                 eps=config['train']['adam_eps'], weight_decay=config['train']['weight_decay'])
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    if optimizer_ids != expected_ids or len(names) == 0:
        raise RuntimeError('Optimizer contains non-local or missing parameters')
    model.train(True)
    optimizer.zero_grad()
    result = model.train_local_batch(images, attributes)
    if result['loss'].dtype != torch.float32 or not torch.isfinite(result['loss']):
        raise FloatingPointError('Invalid FP32 Local BCE loss')
    result['loss'].backward()
    if any(p.grad is not None for n, p in model.named_parameters() if not n.startswith(NEW_PREFIXES)):
        raise RuntimeError('Frozen QueryBalanced parameter received a gradient')
    for n, p in model.named_parameters():
        if n.startswith(NEW_PREFIXES):
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('New module has missing/nonfinite gradient: ' + n)
    optimizer.step()
    digest_after_step = model.old_qb_state_digest()
    if digest_after_step != report['old_qb_state_sha256_before_step']:
        raise AssertionError('Optimizer Step changed a frozen parameter or buffer')
    report['preflight_batch_size'] = batch_size
    report['bce'] = float(result['attr_bce'].detach())
    report['weighted_loss'] = float(result['loss'].detach())
    report['optimizer_parameter_count'] = sum(p.numel() for group in optimizer.param_groups
                                              for p in group['params'])
    report['frozen_gradients_absent'] = True
    report['new_head_gradients_finite'] = True
    report['frozen_state_sha256_after_optimizer_step'] = digest_after_step

    # A full-model raw state_dict round-trip with strict=True, no sidecar state.
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix='frozen_localattr_preflight_', suffix='.pth')
        os.close(fd)
        torch.save(model.state_dict(), temp_path)
        saved = load_state_dict(temp_path)
        if set(saved) != set(model.state_dict()):
            raise RuntimeError('Temporary saved state_dict has mismatched keys')
        model.load_state_dict(saved, strict=True)
        report['temporary_full_state_dict_strict_roundtrip'] = True
        report['temporary_state_dict_bytes'] = Path(temp_path).stat().st_size
        report['temporary_state_dict_sha256'] = file_sha256(temp_path)
    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()
    if device.type == 'cuda':
        torch.cuda.synchronize()
        report['gpu_peak_allocated_mib'] = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        report['gpu_peak_reserved_mib'] = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
    report['elapsed_seconds'] = round(time.time() - started, 2)
    report['passed'] = True
    write_json(args.report, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
