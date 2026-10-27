"""60 complete train+validation epochs; only first ten plain model weights."""
import argparse
import csv
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from solver.lr_scheduler import LRSchedulerWithWarmup
from .config import ModelConfig, TrainConfig, MemoryConfig
from .precision import configure_fp32, assert_model_fp32
from .gradient_cache import optimizer_update
from .checkpoint import save_epoch_weights
from .evaluation import evaluate
from .data import UPARSplit, training_loaders
from .audit import audit_splits
from .io import write_json
from .model import UPARModel
from .tokenizer import PromptTokenizer

METRICS = ('mADM', 'mAP', 'Rank-1', 'Rank-5', 'Rank-10', 'mINP')


def build_optimizer(model, config):
    # Preserve the original one-group Adam, including decay of biases/norms.
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.lr,
                            betas=(config.alpha, config.beta), eps=config.adam_eps,
                            weight_decay=config.weight_decay, foreach=False)


def build_scheduler(optimizer, config):
    return LRSchedulerWithWarmup(optimizer, config.milestones, gamma=config.gamma, mode=config.lrscheduler,
                                warmup_factor=config.warmup_factor, warmup_epochs=config.warmup_epochs,
                                warmup_method=config.warmup_method, total_epochs=config.num_epoch,
                                target_lr=config.target_lr, power=config.power)


def load_config(path):
    value = json.loads(Path(path).read_text())
    if value.get('precision') != 'FP32': raise ValueError('FP32 only')
    model = ModelConfig(**value['model'])
    train = TrainConfig(**value['train'])
    memory = MemoryConfig(**value['memory'])
    canonical = lambda x: json.dumps(x, sort_keys=True)
    if canonical(asdict(train)) != canonical(asdict(TrainConfig())):
        raise ValueError('Keep all baseline TrainConfig values, including 60 epochs and seed 1')
    if canonical(asdict(model)) != canonical(asdict(ModelConfig())):
        raise ValueError('Unsupported model configuration')
    memory.validate(train.batch_size)
    return model, train, memory


def gpu_memory(device):
    if device.type != 'cuda': return {}
    return dict(max_memory_allocated=int(torch.cuda.max_memory_allocated(device)),
                max_memory_reserved=int(torch.cuda.max_memory_reserved(device)),
                memory_allocated=int(torch.cuda.memory_allocated(device)),
                memory_reserved=int(torch.cuda.memory_reserved(device)))


class _GracefulStop(Exception):
    pass


def train(model, loaders, config, memory, device, output, provenance=None, evaluator=evaluate):
    output = Path(output)
    if output.exists(): raise FileExistsError('A new run cannot overwrite an existing directory')
    output.mkdir(parents=True)
    logger = logging.getLogger('ViTL14.' + str(output))
    logger.setLevel(logging.INFO)
    handlers = (logging.FileHandler(str(output / 'train.log')), logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
    stop = [False]
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda signum, frame: stop.__setitem__(0, True))
    status = dict(phase='initializing', pid=os.getpid(), completed_epochs=0, max_epochs=config.num_epoch,
                  global_step=0, precision='FP32', effective_batch_size=config.batch_size,
                  micro_batch_size=memory.micro_batch_size, accumulation_steps=memory.accumulation_steps,
                  sdm_global_batch_size=config.batch_size, gradient_cache=True,
                  checkpoint_policy='plain state_dict only, completed epochs 1..10')
    write_json(output / 'status.json', status)
    train_loader, gallery_loader, query_loader = loaders
    try:
        if len(train_loader) < 1: raise ValueError('Empty training loader')
        memory.validate(config.batch_size)
        model.gradient_checkpointing(memory.gradient_checkpointing)
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        model.to(device).train()
        assert_model_fp32(model)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        write_json(output / 'config.json', dict(model=asdict(model.config), train=asdict(config), memory=asdict(memory),
                                               precision='FP32', sampling=train_loader.sampler.metadata(),
                                               checkpoint_policy=status['checkpoint_policy']))
        if provenance is not None: write_json(output / 'data_audit.json', provenance)
        if hasattr(model, 'pretrained_report'): write_json(output / 'pretrained_verification.json', model.pretrained_report)
        global_step, best_madm = 0, float('-inf')
        logger.info('Fresh official ViT-L/14 FP32; %d epochs, effective batch32, micro%d x%d; exact full-batch SDM cache',
                    config.num_epoch, memory.micro_batch_size, memory.accumulation_steps)
        for epoch in range(1, config.num_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            train_loader.generator.manual_seed(config.seed + epoch - 1)
            model.train()
            epoch_started = time.monotonic()
            samples, totals = 0, dict.fromkeys(('loss', 'sdm_loss', 'map_loss'), 0.0)
            for step, batch in enumerate(train_loader, 1):
                if stop[0]: raise _GracefulStop()
                started = time.monotonic()
                result = optimizer_update(model, optimizer, batch, memory.micro_batch_size)
                global_step += 1
                n = len(batch['images'])
                samples += n
                for key in totals: totals[key] += result[key] * n
                status.update(phase='training', epoch=epoch, batch=step, batches_total=len(train_loader),
                              global_step=global_step, optimizer_steps=global_step,
                              scheduler_steps=epoch - 1, last_loss=result, last_batch_seconds=time.monotonic()-started,
                              learning_rate=optimizer.param_groups[0]['lr'], gpu_memory=gpu_memory(device),
                              updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                write_json(output / 'status.json', status)
                if step <= 3 or step % config.log_period == 0:
                    logger.info('Epoch %d/%d Batch %d/%d LR %.8g %s', epoch, config.num_epoch, step,
                                len(train_loader), scheduler.get_last_lr()[0], json.dumps(result, sort_keys=True))
            if samples != len(train_loader.sampler): raise ValueError('Sampler draw count mismatch')
            if stop[0]: raise _GracefulStop()
            # Release gradients for evaluation, preserving FP32 Adam moments.
            optimizer.zero_grad(set_to_none=True)
            status.update(phase='validation', epoch=epoch)
            write_json(output / 'status.json', status)
            def progress(kind, completed, total):
                if stop[0]: raise _GracefulStop()
                status.update(validation_part=kind, validation_completed=completed, validation_total=total)
                write_json(output / 'status.json', status)
            metrics = evaluator(model, gallery_loader, query_loader, config.eval_query_batch_size,
                                memory.evaluation_micro_batch_size, progress)
            if set(metrics) != set(METRICS) or any(not math.isfinite(metrics[k]) for k in METRICS):
                raise FloatingPointError('Invalid validation metrics')
            lr_used = optimizer.param_groups[0]['lr']
            # Original scheduler unit remains one completed epoch, never a micro batch.
            scheduler.step()
            saved = save_epoch_weights(model, output, epoch)
            best_madm = max(best_madm, metrics['mADM'])
            record = dict(epoch=epoch, global_step=global_step, metrics=metrics,
                          losses={k:v/samples for k,v in totals.items()}, lr=lr_used,
                          next_lr=scheduler.get_last_lr()[0], epoch_seconds=time.monotonic()-epoch_started,
                          weight=str(saved) if saved else None, gpu_memory=gpu_memory(device))
            with (output/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(record, allow_nan=False)+'\n')
            csv_path=output/'metrics.csv'
            existed=csv_path.exists()
            with csv_path.open('a',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=('epoch','global_step',*METRICS,'lr','next_lr','weight'))
                if not existed: writer.writeheader()
                writer.writerow(dict(epoch=epoch,global_step=global_step,**metrics,lr=lr_used,next_lr=scheduler.get_last_lr()[0],weight=str(saved) if saved else ''))
            status.update(phase='epoch_complete', completed_epochs=epoch, metrics=metrics,
                          scheduler_steps=epoch, best_mADM=best_madm, checkpoint=str(saved) if saved else None,
                          gpu_memory=gpu_memory(device))
            write_json(output/'status.json',status)
            logger.info('Validation Epoch %d: %s; weights=%s',epoch,
                        ', '.join('%s=%.3f%%'%(k,metrics[k]*100) for k in METRICS),saved or 'disabled after Epoch10')
            if stop[0]: raise _GracefulStop()
        status.update(phase='complete', finished_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        write_json(output/'status.json',status)
    except _GracefulStop:
        status.update(phase='interrupted', partial_epoch_weights_saved=False,
                      note='No optimizer/scheduler/resume checkpoints permitted by this run policy')
        write_json(output/'status.json',status)
        logger.info('Graceful stop; completed weights preserved; partial epoch not saved')
    except BaseException as error:
        status.update(phase='failed', error=repr(error),gpu_memory=gpu_memory(device))
        write_json(output/'status.json',status)
        logger.exception('Training failed; FP32 will not be lowered')
        raise
    finally:
        for sig, handler in previous.items(): signal.signal(sig,handler)
        for handler in handlers: logger.removeHandler(handler);handler.close()


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description='Fresh 60 Epoch QueryBalanced ViT-L/14 FP32 training')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=root.parent/'data')
    p.add_argument('--pretrained',type=Path,default=root/'weights/ViT-L-14.pt')
    p.add_argument('--config',type=Path,default=root/'configs/train_fp32.json')
    p.add_argument('--gpu-uuid')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();output=args.output.resolve()
    if (root/'experiments') not in output.parents: p.error('Use a new directory inside this project/experiments')
    if output.exists(): raise FileExistsError('Fresh run requires a new output directory')
    if int(os.environ.get('WORLD_SIZE','1')) != 1: raise ValueError('Single GPU only')
    _,config,memory=load_config(args.config)
    configure_fp32();torch.set_num_threads(1)
    if args.device=='cuda':
        from .gpu_guard import configure_gpu
        gpu=configure_gpu(args.gpu_uuid)
        print(json.dumps(dict(event='GPU preflight',**gpu)),flush=True)
    random.seed(config.seed);np.random.seed(config.seed);torch.manual_seed(config.seed)
    if args.device=='cuda':torch.cuda.manual_seed_all(config.seed)
    print('Strictly verifying official pretrained model',flush=True)
    model=UPARModel.from_pretrained(args.pretrained)
    print(json.dumps(model.pretrained_report,sort_keys=True),flush=True)
    tokenizer=PromptTokenizer()
    # Use the original loader, sampler, worker RNG and data joins unchanged.
    loaders=training_loaders(args.data_root,tokenizer,config)
    provenance=audit_splits(loaders[0].dataset.split,loaders[1].dataset.split,args.data_root)
    train(model,loaders,config,memory,torch.device(args.device),output,provenance)


if __name__=='__main__':main()
