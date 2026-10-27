#!/usr/bin/env python3
"""Train only the new multiscale/local modules for exactly five epochs."""
import argparse
import csv
import datetime
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.clip_model import CLIP
from upar.checkpoint import load_state_dict
from upar.data import (GalleryDataset, QueryDataset, TrainingDataset, UPARSplit,
                       validation_loaders)
from upar.frozen_multiscale import FrozenLocalConfig, FrozenMultiScaleLocalModel
from upar.io import write_json
from upar.local_validation import evaluate_local
from scripts.preflight import check_checkpoint_provenance, strict_load_new_model
from upar.pretrained import CLIP_CONFIG
from upar.sampling import QueryBalancedSampler
from upar.tokenizer import PromptTokenizer


KST = datetime.timezone(datetime.timedelta(hours=9))


def log(message, stream):
    line = datetime.datetime.now(KST).isoformat(timespec='seconds') + ' ' + str(message)
    print(line, file=stream, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(expected_uuid):
    command = ['nvidia-smi', '--id=' + expected_uuid,
               '--query-gpu=name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=30, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(',')]
        if values:
            rows.append({'name': values[0], 'uuid': values[1], 'pci_bus_id': values[2],
                         'memory_total_mib': int(values[3]), 'memory_used_mib': int(values[4]),
                         'utilization_gpu_percent': int(values[5])})
    apps_result = subprocess.run(
        ['nvidia-smi', '--id=' + expected_uuid,
         '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        timeout=30, check=True)
    for row in rows:
        row['compute_apps'] = [line.strip() for line in apps_result.stdout.splitlines() if line.strip()]
    return rows


def guard_gpu(preflight, expected_uuid, allow_context=False):
    rows = inventory(expected_uuid)
    if len(rows) != 1 or rows[0]['uuid'] != expected_uuid:
        raise RuntimeError('Expected assigned Mercury GPU %s, found %s' % (expected_uuid, rows))
    gpu = rows[0]
    memory_limit = 1024 if allow_context else 512
    if gpu['memory_used_mib'] > memory_limit or (gpu['compute_apps'] and not allow_context):
        raise RuntimeError('GPU is occupied (%d MiB); leaving every other experiment untouched' %
                           gpu['memory_used_mib'])
    if preflight.get('gpu_uuid') and preflight['gpu_uuid'] != gpu['uuid']:
        raise RuntimeError('GPU UUID differs from the successful preflight')
    return gpu


def save_epoch_weights(model, epoch, run_dir, log_stream):
    if not 1 <= epoch <= 5:
        raise ValueError('Only epochs 1..5 are allowed')
    target = Path(run_dir) / ('epoch_%03d.pth' % epoch)
    descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(model.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    # Strictly reload the entire saved inference state, with source tensors on CPU.
    saved = load_state_dict(target)
    expected = model.state_dict()
    if set(saved) != set(expected):
        raise RuntimeError('Saved Epoch %d model keys do not match the live model' % epoch)
    for name, value in expected.items():
        if saved[name].shape != value.shape or saved[name].dtype != value.dtype:
            raise RuntimeError('Saved Epoch %d shape/dtype mismatch: %s' % (epoch, name))
    model.load_state_dict(saved, strict=True)
    report = {'epoch': epoch, 'path': str(target.resolve()), 'bytes': target.stat().st_size,
              'sha256': sha256_file(target), 'keys': len(saved), 'strict_reload': True,
              'plain_state_dict': True}
    log('Saved and strict-reloaded Epoch %d state_dict (%d bytes, %s)' %
        (epoch, report['bytes'], report['sha256']), log_stream)
    return report


def source_epoch2_metrics(config):
    path = Path(config['initial_run_dir']) / 'metrics.jsonl'
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get('epoch') == 2:
            return row['metrics']
    raise RuntimeError('Initial Epoch-2 validation metrics are missing')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/frozen_localattr_e5.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight-report', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = config['expected_gpu_uuid']
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / 'train.log'
    with log_path.open('a', buffering=1) as log_stream:
        status_path = run_dir / 'status.json'
        metrics_path = run_dir / 'metrics.jsonl'
        manifest_path = run_dir / 'manifest.json'
        try:
            preflight = json.loads(args.preflight_report.read_text())
            if not preflight.get('passed') or preflight.get('device') != 'cuda' \
                    or preflight.get('preflight_batch_size') != 32 \
                    or not preflight.get('temporary_full_state_dict_strict_roundtrip'):
                raise RuntimeError('A complete successful CUDA Batch-32 preflight is required')
            provenance = check_checkpoint_provenance(config)
            if provenance['sha256'] != preflight['initial_checkpoint']['sha256']:
                raise RuntimeError('Source Checkpoint changed after GPU preflight')
            gpu = guard_gpu(preflight, config['expected_gpu_uuid'])
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('CUDA device visibility changed after preflight')
            properties = torch.cuda.get_device_properties(0)
            if properties.name != gpu['name']:
                raise RuntimeError('CUDA runtime and NVML GPU names differ')
            if gpu['memory_total_mib'] < 11000:
                raise RuntimeError('The preflight GPU does not have the expected 12 GiB capacity')

            # The default CUDA mapping is preserved. No GPU is selected with -S or reset.
            device = torch.device('cuda:0')
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            seed = int(config['train']['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # Validate annotation/image paths and build loaders before the
            # 2 GB model transfer, so the GPU is not reserved during storage I/O.
            tokenizer = PromptTokenizer()
            train_split = UPARSplit.read(config['data_root'], 'train')
            val_split = UPARSplit.read(config['data_root'], 'val')
            sampler = QueryBalancedSampler(train_split, seed)
            train_loader = DataLoader(
                TrainingDataset(train_split, tokenizer), batch_size=config['train']['batch_size'],
                sampler=sampler, num_workers=config['train']['num_workers'],
                generator=torch.Generator().manual_seed(seed))
            gallery_loader, query_loader = validation_loaders(
                val_split, tokenizer, config['validation']['test_batch_size'],
                config['validation']['num_workers'])
            baseline_metrics = source_epoch2_metrics(config)
            expected_batch_count = len(train_loader)
            if train_loader.batch_size != 32 or config['train']['gradient_accumulation_steps'] != 1:
                raise RuntimeError('The run must use one physical batch of 32 per optimizer update')
            gpu = guard_gpu(preflight, config['expected_gpu_uuid'], allow_context=True)

            state = load_state_dict(config['initial_checkpoint'])
            if sha256_file(config['initial_checkpoint']) != config['initial_checkpoint_sha256']:
                raise RuntimeError('Epoch-2 source Checkpoint failed its configured SHA256')
            model, added_keys = strict_load_new_model(state, seed)
            del state
            gc.collect()
            gpu = guard_gpu(preflight, config['expected_gpu_uuid'], allow_context=True)
            torch.cuda.manual_seed_all(seed)
            model.to(device)
            model.configure_frozen_training()
            frozen_digest = model.old_qb_state_digest()
            trainable = model.trainable_parameter_names()
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_count = sum(p.numel() for n, p in model.named_parameters()
                               if not n.startswith(('multiscale.', 'local_branch.')))
            optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
            if len(optimizer_parameters) == 0 or len(trainable) == 0:
                raise RuntimeError('No local parameters are trainable')
            optimizer = torch.optim.Adam(optimizer_parameters, lr=config['train']['lr'],
                                         betas=tuple(config['train']['adam_betas']),
                                         eps=config['train']['adam_eps'],
                                         weight_decay=config['train']['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['train']['scheduler_t_max'],
                eta_min=config['train']['scheduler_eta_min'])

            if any(name.startswith(('base_model.', 'cross_attn.', 'cross_modal_transformer.',
                                    'ln_pre_t.', 'ln_pre_i.', 'ln_post.', 'mlm_head.'))
                   for name in trainable):
                raise RuntimeError('Optimizer includes an old QueryBalanced parameter')
            log('Epoch-2 checkpoint strict-loaded: %s' % provenance['path'], log_stream)
            log('GPU %s UUID=%s used=%d/%d MiB' %
                (gpu['name'], gpu['uuid'], gpu['memory_used_mib'], gpu['memory_total_mib']), log_stream)
            log('Parameters frozen=%d trainable=%d; trainable names=%s' %
                (frozen_count, trainable_count, ','.join(trainable)), log_stream)
            log('Objective=0.5 * BCEWithLogits(local_logits, binary_attributes); Adam LR=%.8g; '
                'effective batch=32; accumulation=1; cosine T_max=5, eta_min=1e-5, no warmup' %
                config['train']['lr'], log_stream)

            run_config = dict(config)
            run_config['run_directory'] = str(run_dir)
            run_config['actual_gpu'] = gpu
            run_config['cuda_device_name'] = properties.name
            run_config['cuda_compute_capability'] = list(torch.cuda.get_device_capability(0))
            run_config['initial_checkpoint_provenance'] = provenance
            run_config['added_state_keys'] = added_keys
            run_config['frozen_parameter_count'] = frozen_count
            run_config['trainable_parameter_count'] = trainable_count
            run_config['trainable_parameter_names'] = trainable
            run_config['frozen_state_sha256_before_training'] = frozen_digest
            run_config['effective_batch_count_per_epoch'] = expected_batch_count
            run_config['preflight_report'] = str(args.preflight_report.resolve())
            write_json(run_dir / 'config.json', run_config)
            write_json(manifest_path, {'initial_checkpoint_sha256': provenance['sha256'],
                                       'architecture': config['architecture'],
                                       'frozen_state_sha256_before_training': frozen_digest,
                                       'epochs': [], 'epoch_weight_policy': config['checkpoint_policy']})
            write_json(status_path, {'phase': 'training', 'epoch': 0, 'max_epochs': 5,
                                     'batch': 0, 'batches_total': expected_batch_count,
                                     'pid': os.getpid(), 'device': str(device), 'gpu': gpu,
                                     'precision': 'FP32', 'frozen_state_sha256': frozen_digest,
                                     'frozen_parameter_count': frozen_count,
                                     'trainable_parameter_count': trainable_count,
                                     'effective_batch_size': 32, 'gradient_accumulation_steps': 1,
                                     'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if metrics_path.exists():
                raise FileExistsError(metrics_path)

            start_time = time.time()
            for epoch in range(1, 6):
                if epoch != len([p for p in run_dir.glob('epoch_*.pth')]) + 1:
                    raise RuntimeError('Epoch checkpoint sequence is out of order')
                sampler.set_epoch(epoch)
                model.train(True)
                epoch_loss = 0.0
                epoch_bce = 0.0
                batches_seen = 0
                optimizer.zero_grad(set_to_none=True)
                epoch_started = time.time()
                for batch_index, batch in enumerate(train_loader, 1):
                    images = batch['images'].to(device, non_blocking=True)
                    attributes = batch['attributes'].to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    result = model.train_local_batch(images, attributes)
                    loss = result['loss']
                    if not torch.isfinite(loss) or loss.dtype != torch.float32:
                        raise FloatingPointError('Nonfinite/non-FP32 local BCE at epoch %d batch %d' %
                                                 (epoch, batch_index))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if name.startswith(('multiscale.', 'local_branch.')):
                            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError('Invalid local gradient: ' + name)
                        elif parameter.grad is not None:
                            raise RuntimeError('Frozen base received a gradient: ' + name)
                    optimizer.step()
                    epoch_loss += float(loss.detach())
                    epoch_bce += float(result['attr_bce'].detach())
                    batches_seen += 1
                    if batch_index <= 3 or batch_index % config['train']['log_period'] == 0 \
                            or batch_index == expected_batch_count:
                        lr = optimizer.param_groups[0]['lr']
                        log('Epoch %d/5 Batch %d/%d LR %.8g loss %.6f bce %.6f' %
                            (epoch, batch_index, expected_batch_count, lr,
                             float(loss.detach()), float(result['attr_bce'].detach())), log_stream)
                        write_json(status_path, {'phase': 'training', 'epoch': epoch, 'max_epochs': 5,
                                                 'batch': batch_index, 'batches_total': expected_batch_count,
                                                 'global_step': (epoch - 1) * expected_batch_count + batch_index,
                                                 'last_loss': float(loss.detach()),
                                                 'last_bce': float(result['attr_bce'].detach()),
                                                 'learning_rate': lr, 'pid': os.getpid(), 'device': str(device),
                                                 'gpu': gpu, 'precision': 'FP32',
                                                 'frozen_state_sha256': frozen_digest,
                                                 'updated_at_utc': datetime.datetime.now(
                                                     datetime.timezone.utc).isoformat()})
                if batches_seen != expected_batch_count:
                    raise RuntimeError('Epoch ended after %d of %d batches' %
                                       (batches_seen, expected_batch_count))
                after_epoch_digest = model.old_qb_state_digest()
                if after_epoch_digest != frozen_digest:
                    raise AssertionError('Frozen QueryBalanced parameters/buffers changed during Epoch %d' % epoch)

                model.eval()
                validation_dir = run_dir / ('validation_epoch_%03d' % epoch)

                def update_validation(stage, completed, total):
                    interval = 40 if stage == 'gallery_features' else 500 if stage == 'official_metrics' else 800
                    if completed != total and completed % interval != 0:
                        return
                    write_json(status_path, {'phase': 'validation', 'epoch': epoch,
                                             'max_epochs': 5, 'validation_stage': stage,
                                             'validation_completed': completed,
                                             'validation_total': total,
                                             'completed_epochs': epoch - 1,
                                             'last_training_batch': batches_seen,
                                             'pid': os.getpid(), 'device': str(device), 'gpu': gpu,
                                             'frozen_state_sha256': frozen_digest,
                                             'updated_at_utc': datetime.datetime.now(
                                                 datetime.timezone.utc).isoformat()})
                    log('Validation Epoch %d %s %d/%d' % (epoch, stage, completed, total), log_stream)

                validation = evaluate_local(
                    model, gallery_loader, query_loader, validation_dir,
                    set(train_split.queries), weights=None,
                    micro_batch_size=config['validation']['evaluation_micro_batch_size'],
                    progress=update_validation)
                global_only = validation['methods']['Global-only']
                baseline_delta = {name: global_only[name] - baseline_metrics[name]
                                  for name in baseline_metrics}
                val_row = {'epoch': epoch, 'training': {'weighted_bce': epoch_loss / batches_seen,
                                                        'bce': epoch_bce / batches_seen,
                                                        'batches': batches_seen,
                                                        'epoch_seconds': time.time() - epoch_started},
                           'validation': validation,
                           'global_only_vs_source_epoch2': baseline_delta,
                           'validation_selected_nll_weight': validation['best_weights']['nll'],
                           'learning_rate_used': optimizer.param_groups[0]['lr']}
                with metrics_path.open('a') as stream:
                    stream.write(json.dumps(val_row, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                log('Validation Epoch %d: Global mADM %.6f Local-NLL mADM %.6f '
                    'Fused mADM %.6f (NLL weight=%.2f); global delta=%s' %
                    (epoch, global_only['mADM'], validation['methods']['NLL-only']['mADM'],
                     validation['methods']['Global+NLL']['mADM'], validation['best_weights']['nll'],
                     json.dumps(baseline_delta, sort_keys=True)), log_stream)
                if abs(baseline_delta['mADM']) > 1e-6 or abs(baseline_delta['mAP']) > 1e-6:
                    log('WARNING: global validation metrics differ from the recorded Epoch-2 baseline; '
                        'the frozen state hash and direct score checks remain authoritative', log_stream)

                # The full model state_dict is the only epoch weight artifact.
                saved = save_epoch_weights(model, epoch, run_dir, log_stream)
                if model.old_qb_state_digest() != frozen_digest:
                    raise AssertionError('Strict reload changed frozen QueryBalanced state')
                manifest = json.loads(manifest_path.read_text())
                manifest['epochs'].append(saved)
                write_json(manifest_path, manifest)
                write_json(status_path, {'phase': 'training' if epoch < 5 else 'finalizing',
                                         'epoch': epoch, 'max_epochs': 5,
                                         'batch': expected_batch_count, 'batches_total': expected_batch_count,
                                         'completed_epochs': epoch, 'global_step': epoch * expected_batch_count,
                                         'last_validation': validation['methods'],
                                         'validation_selected_nll_weight': validation['best_weights']['nll'],
                                         'last_checkpoint': saved,
                                         'frozen_state_sha256': frozen_digest,
                                         'pid': os.getpid(), 'device': str(device), 'gpu': gpu,
                                         'precision': 'FP32',
                                         'elapsed_seconds': time.time() - start_time,
                                         'updated_at_utc': datetime.datetime.now(
                                             datetime.timezone.utc).isoformat()})
                # Advance once after the completed validation and checkpoint.
                scheduler.step()

            expected = {run_dir / ('epoch_%03d.pth' % epoch) for epoch in range(1, 6)}
            actual = set(run_dir.glob('epoch_*.pth'))
            if actual != expected:
                raise RuntimeError('Expected exactly Epoch 1..5 checkpoints; found %s' % sorted(actual))
            final_digest = model.old_qb_state_digest()
            if final_digest != frozen_digest:
                raise AssertionError('Frozen QueryBalanced state changed before normal shutdown')
            if len(json.loads(manifest_path.read_text())['epochs']) != 5:
                raise RuntimeError('Manifest does not contain five completed epoch weights')
            del optimizer, scheduler, model, train_loader, gallery_loader, query_loader
            gc.collect()
            torch.cuda.synchronize()
            peak_allocated = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            peak_reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 2, 2)
            torch.cuda.empty_cache()
            write_json(status_path, {'phase': 'complete', 'completed_epochs': 5, 'max_epochs': 5,
                                     'checkpoints': [str(p.resolve()) for p in sorted(expected)],
                                     'frozen_state_sha256_before_training': frozen_digest,
                                     'frozen_state_sha256_after_training': final_digest,
                                     'frozen_state_unchanged': True,
                                     'gpu': gpu, 'gpu_peak_allocated_mib': peak_allocated,
                                     'gpu_peak_reserved_mib': peak_reserved,
                                     'cuda_cache_released': True, 'pid': os.getpid(),
                                     'elapsed_seconds': time.time() - start_time,
                                     'updated_at_utc': datetime.datetime.now(
                                         datetime.timezone.utc).isoformat()})
            log('Normal completion after exactly five epochs; CUDA cache released; no Epoch 6 started.',
                log_stream)
        except Exception as error:
            write_json(status_path, {'phase': 'failed', 'pid': os.getpid(),
                                     'error': repr(error),
                                     'traceback': __import__('traceback').format_exc(),
                                     'updated_at_utc': datetime.datetime.now(
                                         datetime.timezone.utc).isoformat()})
            log('FAILED: %r' % error, log_stream)
            raise


if __name__ == '__main__':
    run()
