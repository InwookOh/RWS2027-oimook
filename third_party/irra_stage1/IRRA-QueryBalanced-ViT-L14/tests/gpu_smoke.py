"""Actual Mercury GPU, actual QueryBalanced batches, FP32 gradients and Adam.

This bounded test never resumes an experiment or creates a training run.
"""
import argparse
from dataclasses import replace
import gc
from pathlib import Path
import random
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from upar.config import TrainConfig
from upar.data import UPARSplit, TrainingDataset, validation_loaders
from upar.sampling import QueryBalancedSampler
from upar.tokenizer import PromptTokenizer
from upar.audit import audit_splits
from upar.model import UPARModel
from upar.training import build_optimizer, build_scheduler, load_config, gpu_memory
from upar.gradient_cache import optimizer_update
from upar.precision import configure_fp32, assert_model_fp32, assert_optimizer_fp32
from upar.evaluation import evaluate
from upar.checkpoint import save_epoch_weights, load_state_dict
from upar.gpu_guard import configure_gpu
from upar.io import write_json


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=root/'configs/train_fp32.json')
    p.add_argument('--gpu-uuid')
    p.add_argument('--steps',type=int,default=2)
    p.add_argument('--report-name',default='gpu_smoke')
    args=p.parse_args()
    if not args.report_name.replace('_','').isalnum(): p.error('Invalid report basename')
    if args.steps<2: p.error('Use at least two updates to inspect stable allocation')
    started=time.monotonic();configure_fp32();torch.set_num_threads(1)
    gpu=configure_gpu(args.gpu_uuid)
    report=dict(passed=False,device='cuda:0',gpu=gpu,precision='FP32',full_training_started=False,
                actual_dataset=True,steps=[],config=str(args.config))
    write_json(root/'reports'/(args.report_name+'_status.json'),dict(phase='loading',**report))
    try:
        random.seed(1);np.random.seed(1);torch.manual_seed(1);torch.cuda.manual_seed_all(1)
        _,config,memory=load_config(args.config)
        print('GPU PREFLIGHT',gpu,flush=True)
        print('Loading and strictly verifying official ViT-L/14',flush=True)
        model=UPARModel.from_pretrained(root/'weights/ViT-L-14.pt')
        report['pretrained']=model.pretrained_report
        report['parameters']=sum(p.numel() for p in model.parameters())
        report['architecture']=dict(vision_blocks=len(model.base_model.visual.transformer.resblocks),
                                    vision_width=model.base_model.visual.conv1.out_channels,
                                    vision_heads=model.base_model.visual.transformer.resblocks[0].attn.num_heads,
                                    text_blocks=len(model.base_model.transformer.resblocks),
                                    text_width=model.base_model.token_embedding.embedding_dim,
                                    text_heads=model.base_model.transformer.resblocks[0].attn.num_heads,
                                    projection=list(model.base_model.text_projection.shape),
                                    position=list(model.base_model.visual.positional_embedding.shape),
                                    MAP_dim=model.embed_dim,MAP_heads=model.cross_attn.num_heads,
                                    MAP_blocks=len(model.cross_modal_transformer.resblocks))
        tokenizer=PromptTokenizer()
        print('Reading actual train/validation and testing QueryBalanced sampler',flush=True)
        train=UPARSplit.read(root.parent/'data','train');val=UPARSplit.read(root.parent/'data','val')
        audit=audit_splits(train,val,root.parent/'data')
        report['data_audit']=audit
        write_json(root/'reports/data_audit.json',audit)
        sampler=QueryBalancedSampler(train,1);sampler.set_epoch(1)
        loader=DataLoader(TrainingDataset(train,tokenizer),batch_size=32,sampler=sampler,num_workers=0,
                          generator=torch.Generator().manual_seed(1))
        iterator=iter(loader)
        torch.cuda.reset_peak_memory_stats()
        model.gradient_checkpointing(memory.gradient_checkpointing);model.cuda().train()
        optimizer=build_optimizer(model,config);scheduler=build_scheduler(optimizer,config)
        names=['base_model.visual.conv1.weight','base_model.text_projection','cross_attn.in_proj_weight']
        parameters=dict(model.named_parameters())
        before={name:parameters[name].detach().cpu().clone() for name in names}
        for index in range(args.steps):
            batch=next(iterator);assert len(batch['images'])==32
            for key,value in batch.items():
                if value.is_floating_point(): assert value.dtype==torch.float32
            write_json(root/'reports'/(args.report_name+'_status.json'),dict(phase='update',step=index+1,gpu=gpu))
            begin=time.monotonic()
            losses=optimizer_update(model,optimizer,batch,memory.micro_batch_size)
            torch.cuda.synchronize()
            assert_model_fp32(model,gradients=True);assert_optimizer_fp32(optimizer)
            assert losses['sdm_batch_size']==32 and losses['effective_batch_size']==32
            assert scheduler.last_epoch==0  # no iteration/micro-batch scheduler step
            snapshot=dict(step=index+1,seconds=time.monotonic()-begin,losses=losses,**gpu_memory(torch.device('cuda')))
            report['steps'].append(snapshot)
            print('ACTUAL FP32 UPDATE',snapshot,flush=True)
            write_json(root/'reports'/(args.report_name+'_progress.json'),report)
        changes={name:float((parameters[name].detach().cpu()-before[name]).abs().max()) for name in names}
        if not all(value>0 for value in changes.values()): raise AssertionError('Weights did not update: '+str(changes))
        report['representative_weight_max_changes']=changes
        report['both_encoders_receive_gradients']=all(parameters[name].grad is not None for name in names[:2])
        allocated=[r['memory_allocated'] for r in report['steps']]
        report['steady_state_allocation_delta_bytes']=max(allocated)-min(allocated)
        if report['steady_state_allocation_delta_bytes']>32*1024**2: raise AssertionError('Unexpected steady-state growth')
        optimizer.zero_grad(set_to_none=True)
        # Genuine val samples, all query labels derived from the same subset gallery.
        # This verifies the metric path; it is not a full validation score.
        records=val.images[:32]
        used={r.query_index for r in records}
        queries=tuple(q for i,q in enumerate(val.queries) if i in used)
        subset=replace(val,images=records,queries=queries)
        gallery,query=validation_loaders(subset,tokenizer,batch_size=32,num_workers=0)
        print('Actual small public-validation retrieval check',flush=True)
        metrics=evaluate(model,gallery,query,config.eval_query_batch_size,memory.evaluation_micro_batch_size)
        report['validation_subset']=dict(gallery=len(records),queries=len(queries),metrics=metrics,full_validation=False)
        with tempfile.TemporaryDirectory(prefix='vitl14-weights-smoke-') as temporary:
            print('Plain state_dict save/reload integrity check in temporary folder',flush=True)
            path=save_epoch_weights(model,temporary,1)
            state=load_state_dict(path)
            assert set(state)==set(model.state_dict())
            assert all(torch.is_tensor(v) and v.dtype==torch.float32 for v in state.values())
            assert all(torch.equal(value.cpu(),state[name]) for name,value in model.state_dict().items())
            report['plain_state_dict_round_trip']=True
            report['temporary_weight_bytes']=path.stat().st_size
            # Keep only CPU state while releasing the trained smoke model/optimizer.
            del parameters,optimizer,model,before,iterator,loader;gc.collect();torch.cuda.empty_cache()
            restored=UPARModel.from_state_dict(state,with_map=False).eval()
            assert_model_fp32(restored)
            report['retrieval_strict_reload']=True
            del restored,state
        report.update(passed=True,oom=False,all_model_and_gradient_FP32=True,optimizer_state_FP32=True,
                      effective_batch_size=32,micro_batch_size=memory.micro_batch_size,
                      accumulation_steps=memory.accumulation_steps,sdm_global_batch=32,
                      gradient_checkpointing=memory.gradient_checkpointing,gradient_cache=True,
                      seconds=time.monotonic()-started,peak=gpu_memory(torch.device('cuda')))
        write_json(root/'reports'/(args.report_name+'.json'),report)
        write_json(root/'reports'/(args.report_name+'_status.json'),dict(phase='complete',passed=True))
        print('ACTUAL GPU CHECK PASSED',report['seconds'],flush=True)
    except BaseException as error:
        report.update(error=repr(error),oom='out of memory' in str(error).lower(),peak=gpu_memory(torch.device('cuda')),
                      seconds=time.monotonic()-started)
        write_json(root/'reports'/(args.report_name+'.json'),report)
        write_json(root/'reports'/(args.report_name+'_status.json'),dict(phase='failed',error=repr(error)))
        raise


if __name__=='__main__':main()
