"""Train only on the official train split; select checkpoints by validation mADM."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import subprocess

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from upar.attributes import ATTRIBUTES
from upar.results import save_predictions, append_results
from upar.config import LOSSES, normalize_config, model_options
from upar.data import DOMAINS, ImageDataset, load_split, make_transform, select_domains
from upar.evaluation import evaluate_split
from upar.experiments import DeepMARLoss, ModelEMA, smooth_targets
from upar.model import ARCHITECTURES, build_model
from upar.pretrained import SOURCES as PRETRAINED_SOURCES, fetch as fetch_pretrained, load_pretrained_backbone
from upar.retrieval import predict_probabilities
from upar.text_prior import clip_embeddings, initialize_head


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def save_atomic(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_training_state(model, optimizer, gradient_clip):
    """Reject corrupted parameters/moments before accepting a checkpoint."""
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter.detach()).all():
            raise RuntimeError(f'Non-finite model parameter: {name}')
        state = optimizer.state.get(parameter, {}) if optimizer is not None else {}
        for key, value in state.items():
            if torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    raise RuntimeError(f'Non-finite optimizer state: {name}/{key}')
                # With clipped global L2 norm, each squared gradient is bounded.
                if key == 'exp_avg_sq' and (value.min() < 0 or value.max() > gradient_clip ** 2 * 1.001):
                    raise RuntimeError(f'Optimizer second moment outside gradient-clipping bound: {name}')


def inference_checkpoint(model, config, epoch, metric):
    return {
        'format_version': 1, 'architecture': config['architecture'],
        'attribute_names': list(ATTRIBUTES), 'preprocess': config['preprocess'],
        'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'inference_batch_size': config['inference_batch_size'],
        'query_chunk_size': config['query_chunk_size'],
        'epoch': epoch, 'validation_mADM': metric,
        'training_config': config,
        'weight_source': 'ema' if config.get('ema', False) else 'model',
    }


def git_commit(root):
    """HEAD commit; reads .git directly where the git binary is unavailable (e.g. Kubernetes pods)."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    git = Path(root) / '.git'
    try:
        head = (git / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def record_run_metadata(output, config, data_dir, device, resume):
    root = Path(__file__).resolve().parent
    def sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    source_paths = [root / name for name in ('train.py', 'evaluate.py', 'build_submission.py', 'verify_submission.py')]
    source_paths += sorted((root / 'upar').glob('*.py'))
    annotations = [Path(data_dir) / 'annotations/task2' / split / name
                   for split in ('train', 'val') for name in ('gt.csv', 'ids.csv', 'queries.csv')]
    commit = git_commit(root)
    record = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': commit,
        'source_sha256': {str(p.relative_to(root)): sha256(p) for p in source_paths},
        'annotation_sha256': {str(p): sha256(p) for p in annotations},
        'config': config, 'resume': resume, 'device': str(device),
        'python': sys.version.split()[0], 'torch': str(torch.__version__),
        'torchvision': str(torchvision.__version__), 'numpy': np.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
    }
    with (output / 'run_metadata.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs/baseline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', help='Local last.training.pt from this code (trusted file only)')
    parser.add_argument('--backbone', '--architecture', dest='architecture', choices=ARCHITECTURES)
    parser.add_argument('--pretrained', choices=('none', *PRETRAINED_SOURCES))
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--holdout-domain', choices=DOMAINS,
                        help='Leave-one-domain-out: exclude this domain from training/selection and report it separately')
    parser.add_argument('--max-train-images', type=int, help='Smoke test only; first N rows')
    parser.add_argument('--max-val-images', type=int, help='Smoke test only; first N rows')
    args = parser.parse_args()
    resume = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    config = dict(resume['config'] if resume else json.loads(Path(args.config).read_text()))
    for key in ('epochs', 'batch_size', 'workers'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if not resume:
        for key in ('architecture', 'pretrained', 'holdout_domain'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.architecture:
            config['backbone'] = args.architecture
    elif args.architecture or args.pretrained or args.holdout_domain:
        parser.error('Do not change architecture/pretraining/holdout domain when resuming')
    if args.amp is not None:
        if resume and args.amp != config.get('amp', False):
            parser.error('Keep the original AMP setting when resuming')
        config['amp'] = args.amp
    config = normalize_config(config)
    if config['pretrained'] not in ('none', *PRETRAINED_SOURCES):
        raise ValueError(f'pretrained must be none or one of {tuple(PRETRAINED_SOURCES)}')
    if config['loss'] not in LOSSES:
        raise ValueError(f'Loss must be one of {LOSSES}')
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'inference_batch_size', 'query_chunk_size', 'cpu_threads'):
        if config[key] < 1:
            raise ValueError(f'{key} must be positive')
    if config['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    config.setdefault('optimizer_foreach', False)
    if not np.isfinite(config['gradient_clip']) or config['gradient_clip'] <= 0:
        raise ValueError('gradient_clip must be finite and positive')
    for key in ('max_train_images', 'max_val_images'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
        else:
            config.setdefault(key, None)
    output = Path(args.output_dir)
    if resume and output.resolve() != Path(args.resume).resolve().parent:
        raise ValueError('Resume into the original output directory so its best checkpoint/history are preserved')
    if not resume and (output / 'last.training.pt').exists():
        raise ValueError('Output already contains a run; use --resume or a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    torch.set_num_threads(config['cpu_threads'])
    deterministic = bool(config.get('deterministic', True))
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    # Cross-device bitwise equality is not promised. Benchmarking picks varying cuDNN algorithms,
    # so it is enabled only for non-deterministic runs (about 1.5x faster for ConvNeXt-B on TITAN X).
    torch.backends.cudnn.benchmark = not deterministic
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; inspect the driver or use --device cpu')
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    record_run_metadata(output, config, args.data_dir, device, args.resume)
    train_names, train_labels, _, _ = load_split(args.data_dir, 'train', config['max_train_images'])
    val_names, val_labels, queries, val_ids = load_split(args.data_dir, 'val', config['max_val_images'])
    holdout = config['holdout_domain']
    if holdout:
        # Training and checkpoint selection never see the held-out domain, as for the hidden test domain.
        seen = [d for d in DOMAINS if d != holdout]
        holdout_names, holdout_labels, holdout_ids = select_domains(val_names, val_labels, val_ids, [holdout])
        train_names, train_labels, _ = select_domains(train_names, train_labels, np.zeros(len(train_names)), seen)
        val_names, val_labels, val_ids = select_domains(val_names, val_labels, val_ids, seen)
    if config['train_with_val']:
        # Train on train + (seen-domain) validation images. Validation scores are then in-sample
        # and only logged; choose epochs with LODO instead.
        train_names, train_labels = train_names + val_names, np.concatenate([train_labels, val_labels])
        print('Training on train + validation images; validation scores are in-sample', flush=True)
    root = Path(args.data_dir).resolve()
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(
        ImageDataset([str(root / p) for p in train_names], make_transform(config['preprocess'], True, config['augmix']),
                     train_labels.astype(np.float32)),
        batch_size=config['batch_size'], shuffle=True, num_workers=config['workers'],
        worker_init_fn=seed_worker, generator=generator, pin_memory=device.type == 'cuda',
    )
    # Guard: training images must be normalized exactly like validation and inference images.
    train_steps, eval_steps = loader.dataset.transform.transforms, make_transform(config['preprocess']).transforms
    if (type(train_steps[-1]).__name__ != 'Normalize' or train_steps[-1].mean != eval_steps[-1].mean
            or train_steps[-1].std != eval_steps[-1].std):
        raise RuntimeError('Training transform does not end with the inference normalization')
    model = build_model(config['backbone'], **model_options(config))
    if config['pretrained'] in PRETRAINED_SOURCES and resume is None:
        load_pretrained_backbone(model, config['pretrained'], config['backbone'])
    model = model.to(device)
    if config['head_init'] == 'text_prior' and resume is None:
        # Head from CLIP text plus a regularized residual probe, fitted on frozen features of
        # seen-domain train images only (upar/text_prior.py); replaces the linear-probe epoch.
        started = time.monotonic()
        rng = np.random.default_rng(config['seed'])
        domains = np.asarray([p.split('/')[0] for p in train_names])
        sample = np.sort(np.concatenate([rng.choice(np.flatnonzero(domains == d), min(config['head_init_images_per_domain'],
                                                                                      int((domains == d).sum())), replace=False)
                                         for d in sorted(set(domains))]))
        state = torch.load(fetch_pretrained(config['pretrained']), map_location='cpu', weights_only=True)
        embeddings = clip_embeddings(model, state['visual.proj'], [str(root / train_names[i]) for i in sample],
                                     config['preprocess'], device, config['inference_batch_size'], config['workers'],
                                     bool(config['amp'] and device.type == 'cuda'))
        initialize_head(model, state, embeddings, train_labels[sample], config['head_init_l2'], config['pretrained'])
        del state
        print(f'Text-prior head fitted on {len(sample)} train images of {sorted(set(domains))} '
              f'({time.monotonic() - started:.0f}s)', flush=True)
        if config['save_epoch_checkpoints']:
            # Epoch 0 = pretrained backbone + text-prior head: the WiSE-FT base for scripts/soup_lodo.py.
            save_atomic(inference_checkpoint(model, config, 0, None), output / 'epoch0.pt')
        if holdout:
            probabilities = predict_probabilities(model, [str(root / p) for p in holdout_names], config['preprocess'],
                                                  device, config['inference_batch_size'], config['workers'], amp=bool(config['amp'] and device.type == 'cuda'))
            if config['prediction_cache']:
                save_predictions(output / 'holdout_predictions_epoch0.npz', probabilities, holdout_labels, queries,
                                 holdout_ids, holdout_names, {'epoch': 0, 'seed': config['seed'], 'config': config,
                                                              'holdout_domain': holdout, 'weight_source': 'model'})
            report = evaluate_split(queries, holdout_labels, holdout_ids, probabilities, holdout_names,
                                    config['query_chunk_size'], by_domain=False, retrieval='bce')
            print(f'Epoch 0 (text-prior head): holdout {holdout} mADM={report["metrics"]["mADM"]:.4f} (BCE)', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'],
                                  foreach=config['optimizer_foreach'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=4)
    pos_weight = None
    if config['loss'] == 'weighted_bce':
        positive = train_labels.sum(0)
        pos_weight = torch.tensor(np.clip((len(train_labels) - positive) / np.maximum(positive, 1), 0.1, 10), device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if config['loss'] == 'deepmar_bce':
        criterion = DeepMARLoss(train_labels.mean(0)).to(device)
    use_amp = bool(config['amp'] and device.type == 'cuda')
    # Validation/holdout passes use the same float16 autocast (probabilities differ by ~3e-4 on average).
    eval_amp = use_amp
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    start_epoch, best = 1, -1.0
    if resume:
        model.load_state_dict(resume['model_state'], strict=True)
        optimizer.load_state_dict(resume['optimizer_state'])
        for group in optimizer.param_groups:
            group['foreach'] = config['optimizer_foreach']
        validate_training_state(model, optimizer, config['gradient_clip'])
        scheduler.load_state_dict(resume['scheduler_state'])
        scaler.load_state_dict(resume['scaler_state'])
        start_epoch, best = resume['epoch'] + 1, resume['best_mADM']
        random.setstate(resume['python_rng']); np.random.set_state(resume['numpy_rng'])
        torch.set_rng_state(resume['torch_rng']); generator.set_state(resume['loader_rng'])
        if device.type == 'cuda' and resume['cuda_rng'] is not None:
            torch.cuda.set_rng_state(resume['cuda_rng'], device)
    print(f'Run: {config["architecture"]}, device={device}, train={len(train_names)}, val={len(val_names)}'
          + (f', holdout={holdout} ({len(holdout_names)} images)' if holdout else ''), flush=True)
    print(f'Python={sys.version.split()[0]}, PyTorch={torch.__version__}; full config saved in {output / "config.json"}', flush=True)
    if config['max_train_images'] or config['max_val_images']:
        print('SMOKE/SUBSET RUN: these metrics are not full-validation results.', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    ema = ModelEMA(model, config['ema_decay']) if config['ema'] else None
    if ema is not None and resume:
        if resume.get('ema_state') is None:
            raise ValueError('EMA-enabled resume requires a saved EMA state')
        ema.model.load_state_dict(resume['ema_state'], strict=True)
        ema.num_updates = resume['ema_num_updates']
        validate_training_state(ema.model, None, config['gradient_clip'])
    for epoch in range(start_epoch, config['epochs'] + 1):
        started = time.monotonic()
        last_progress = started
        # LP-FT (Kumar et al., ICLR 2022): train only the new classifier first so that the
        # random head does not distort pretrained features, then fine-tune everything.
        probing = epoch <= config['linear_probe_epochs']
        # With a fitted text-prior CLS head, probing warms up only the (zero-initialized) query
        # head: the CLS head and its dropout stay fixed so the query branch learns a residual.
        probe_prefix = 'head_query.' if config['head_init'] == 'text_prior' else model.head_prefix
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(not probing or name.startswith(probe_prefix))
        if hasattr(model, 'head_drop'):
            model.head_drop.p = 0.0 if probing and config['head_init'] == 'text_prior' else config['dropout']
        if probing or epoch == config['linear_probe_epochs'] + 1 > 1:
            for group in optimizer.param_groups:
                group['lr'] = config['linear_probe_lr'] if probing else config['learning_rate']
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss, total_images = 0.0, 0
        accumulation = config['accumulation_steps']
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            group_start = (step // accumulation) * accumulation
            group_batches = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(images), smooth_targets(labels, config['label_smoothing']))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}, batch {step + 1}')
            # Weight by actual image count, including a short final batch.
            group_images = min(accumulation * config['batch_size'], len(train_names) - group_start * config['batch_size'])
            scaler.scale(loss * len(images) / group_images).backward()
            if (step - group_start + 1) == group_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'],
                                               error_if_nonfinite=not use_amp, foreach=False)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if ema is not None and scaler.get_scale() >= previous_scale:
                    ema.update(model)
                if (step + 1) % 100 == 0 or step + 1 == len(loader):
                    validate_training_state(model, optimizer, config['gradient_clip'])
            total_loss += loss.item() * len(images); total_images += len(images)
            now = time.monotonic()
            if step == 0 or (step + 1) % 100 == 0 or step + 1 == len(loader) or now - last_progress >= 30:
                elapsed = now - started
                remaining = elapsed / (step + 1) * (len(loader) - step - 1)
                print(f'Epoch {epoch}/{config["epochs"]} batch {step + 1}/{len(loader)} '
                      f'loss={total_loss / total_images:.5f} elapsed={elapsed / 60:.1f}m '
                      f'train_eta={remaining / 60:.1f}m', flush=True)
                last_progress = now
        training_finished = time.monotonic()
        evaluation_model = ema.model if ema is not None else model
        probabilities = predict_probabilities(evaluation_model, [str(root / p) for p in val_names], config['preprocess'], device,
                                              config['inference_batch_size'], config['workers'], amp=eval_amp)
        gallery_finished = time.monotonic()
        cache_path = output / 'val_predictions.npz'
        if config['prediction_cache']:
            save_predictions(cache_path, probabilities, val_labels, queries, val_ids, val_names,
                             {'epoch': epoch, 'seed': config['seed'], 'amp': use_amp,
                              'config': config, 'subset_run': bool(config['max_val_images']),
                              'weight_source': 'ema' if ema is not None else 'model'})
        report = evaluate_split(queries, val_labels, val_ids, probabilities, val_names, config['query_chunk_size'],
                                by_domain=False, retrieval=config['retrieval'])
        holdout_report = None
        if holdout:
            holdout_probabilities = predict_probabilities(
                evaluation_model, [str(root / p) for p in holdout_names], config['preprocess'], device,
                config['inference_batch_size'], config['workers'], amp=eval_amp)
            if config['prediction_cache']:
                # Kept per epoch so retrieval variants can be compared on the unseen domain without retraining.
                save_predictions(output / f'holdout_predictions_epoch{epoch}.npz', holdout_probabilities,
                                 holdout_labels, queries, holdout_ids, holdout_names,
                                 {'epoch': epoch, 'seed': config['seed'], 'amp': use_amp, 'config': config,
                                  'subset_run': bool(config['max_val_images']), 'holdout_domain': holdout,
                                  'weight_source': 'ema' if ema is not None else 'model'})
            holdout_report = evaluate_split(queries, holdout_labels, holdout_ids, holdout_probabilities, holdout_names,
                                            config['query_chunk_size'], by_domain=False, retrieval=config['retrieval'])
        metrics_finished = time.monotonic()
        score = report['metrics']['mADM']
        scheduler.step(score)
        validate_training_state(model, optimizer, config['gradient_clip'])
        if ema is not None:
            validate_training_state(ema.model, None, config['gradient_clip'])
        artifact = inference_checkpoint(evaluation_model, config, epoch, score)
        if config['save_epoch_checkpoints']:
            # Lets LODO-derived stopping rules pick an epoch other than the seen-domain best.
            save_atomic(artifact, output / f'epoch{epoch}.pt')
        if score > best:
            best = score
            save_atomic(artifact, output / 'best.pt')
            if config['prediction_cache']:
                shutil.copyfile(cache_path, output / 'best_val_predictions.npz')
        save_atomic({
            **artifact, 'config': config, 'best_mADM': best,
            # Resume live training weights; best.pt may instead contain EMA weights.
            'model_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'ema_state': {k: v.detach().cpu() for k, v in ema.model.state_dict().items()} if ema is not None else None,
            'ema_num_updates': ema.num_updates if ema is not None else 0,
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(), 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'loader_rng': generator.get_state(),
            'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
        }, output / 'last.training.pt')
        finished = time.monotonic()
        record = {'epoch': epoch, 'train_loss': total_loss / total_images, 'seconds': finished - started,
                  'timings': {'training_seconds': training_finished - started,
                              'gallery_inference_seconds': gallery_finished - training_finished,
                              'retrieval_metrics_seconds': metrics_finished - gallery_finished,
                              'checkpoint_and_state_validation_seconds': finished - metrics_finished},
                  'learning_rate': optimizer.param_groups[0]['lr'], 'validation': report,
                  'phase': 'linear_probe' if probing else 'finetune',
                  'holdout_domain': holdout, 'holdout': holdout_report,
                  'best_mADM': best,
                  'peak_gpu_gib': torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None}
        with (output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        append_results(output / 'results.csv', report,
                       {'run': str(output.resolve()), 'epoch': epoch, 'seed': config['seed'],
                        'amp': use_amp, 'retrieval': config['retrieval'],
                        'subset_run': bool(config['max_val_images']),
                        'cache': str(cache_path) if config['prediction_cache'] else '',
                        'training_seconds': record['timings']['training_seconds'],
                        'gallery_inference_seconds': record['timings']['gallery_inference_seconds'],
                        'metrics_seconds': record['timings']['retrieval_metrics_seconds']})
        if holdout_report is not None:
            append_results(output / 'holdout_results.csv', holdout_report,
                           {'run': str(output.resolve()), 'epoch': epoch, 'seed': config['seed'], 'amp': use_amp,
                            'retrieval': config['retrieval'], 'subset_run': bool(config['max_val_images'])})
            print(f'Epoch {epoch}: seen-domain val mADM={score:.4f}, '
                  f'holdout {holdout} mADM={holdout_report["metrics"]["mADM"]:.4f}', flush=True)
        print(json.dumps(record), flush=True)
    print(f'Finished. Best validation mADM={best:.6f}; checkpoint={output / "best.pt"}', flush=True)


if __name__ == '__main__':
    main()
