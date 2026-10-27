"""Independent CPU numerical and checkpoint-policy tests."""
import ast
import copy
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.objectives import compute_attribute_sdm, compute_map
from model.recompute import recompute_block
from upar.config import ModelConfig, TrainConfig, MemoryConfig
from upar.gradient_cache import backward_effective_batch, optimizer_update
from upar.precision import configure_fp32, assert_model_fp32
from upar.training import build_optimizer, build_scheduler, train, load_config
from upar.checkpoint import save_epoch_weights, load_state_dict
from upar.data import UPARSplit, ImageRecord
from upar.sampling import QueryBalancedSampler
from upar.audit import audit_splits
from upar.evaluation import evaluate
from upar.metrics import retrieval_metrics
from upar.pretrained import resize_visual_position
from upar.prompts import encode_query
from upar.tokenizer import PromptTokenizer

ROOT=Path(__file__).resolve().parents[1]
COMPARISONS=[]


def attrs(i):
    values=[0]*40;values[i%3]=1;return tuple(values)


def sample_batch(n):
    generator=torch.Generator().manual_seed(100+n)
    tokens=torch.randint(1,31,(n,4),generator=generator)
    masked=tokens.clone();labels=torch.full_like(tokens,-100)
    for i in range(n):
        labels[i,:i%5]=tokens[i,:i%5];masked[i,:i%5]=0
    return dict(images=torch.randn(n,6,generator=generator),attributes=torch.tensor([attrs(i) for i in range(n)]),
                caption_ids=tokens,map_ids=masked,map_labels=labels)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Linear(6,8);self.text=nn.Embedding(32,8)
        self.shared=nn.Sequential(nn.Linear(8,8),nn.GELU(),nn.LayerNorm(8))
        self.fusion=nn.Linear(16,8);self.head=nn.Linear(8,32)
        self.register_buffer('logit_scale',torch.tensor(5.0))
        self.config=ModelConfig();self.checkpointing=False
    def gradient_checkpointing(self,flag):self.checkpointing=flag
    def transform(self,x):return recompute_block(self.shared,x) if self.checkpointing else self.shared(x)
    def global_features(self,batch):
        image=self.transform(self.image(batch['images']))
        text=self.transform(self.text(batch['caption_ids']).mean(1))
        return image,text,image[:,None,:]
    def map_result(self,batch,tokens):
        text=self.transform(self.text(batch['map_ids']))
        image=tokens.expand(-1,text.shape[1],-1)
        fused=self.fusion(torch.cat((text,image),-1)).tanh()
        selected=batch['map_labels']!=-100;scores=self.head(fused[selected]);labels=batch['map_labels'][selected]
        return dict(map_loss=compute_map(scores,labels),map_tokens=selected.sum(),
                    map_correct=scores.argmax(-1).eq(labels).sum() if len(labels) else selected.sum()*0)
    def forward(self,batch):
        image,text,tokens=self.global_features(batch);result=self.map_result(batch,tokens)
        result['sdm_loss']=compute_attribute_sdm(image,text,batch['attributes'],self.logit_scale)
        return result
    def encode_image(self,x):return self.transform(self.image(x))
    def encode_text(self,x):return self.transform(self.text(x).mean(1))


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure_fp32();torch.set_num_threads(1)
    def setUp(self):
        torch.manual_seed(1)
        self.tmp=tempfile.TemporaryDirectory(prefix='qb-vitl14-cpu-')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_gradient_cache_matches_monolithic_both_encoders_and_MAP(self):
        for n in (32,5):
            for micro in (1,2,4):
                for checkpoint in (False,True):
                    with self.subTest(n=n,micro=micro,checkpoint=checkpoint):
                        full=Tiny().train();full.gradient_checkpointing(checkpoint)
                        cached=copy.deepcopy(full);data=sample_batch(n)
                        losses=full(data);loss=losses['sdm_loss']+losses['map_loss'];loss.backward()
                        got=backward_effective_batch(cached,data,micro)
                        self.assertAlmostEqual(got['loss'],float(loss),places=5)
                        self.assertEqual(got['sdm_batch_size'],n)
                        errors=[]
                        for (name,p),(_,q) in zip(full.named_parameters(),cached.named_parameters()):
                            self.assertIsNotNone(p.grad,name);self.assertIsNotNone(q.grad,name)
                            self.assertEqual(q.grad.dtype,torch.float32)
                            torch.testing.assert_close(p.grad,q.grad,atol=5e-6,rtol=5e-5,msg=name)
                            errors.append(float((p.grad-q.grad).abs().max()))
                        self.assertGreater(float(cached.image.weight.grad.norm()),0)
                        self.assertGreater(float(cached.text.weight.grad.norm()),0)
                        COMPARISONS.append(dict(effective_batch=n,micro=micro,checkpointing=checkpoint,
                                                max_absolute_gradient_error=max(errors)))

    def test_empty_MAP_supervision_keeps_SDM_backward(self):
        data=sample_batch(5);data['map_labels'].fill_(-100)
        result=backward_effective_batch(Tiny().train(),data,1)
        self.assertEqual(result['map_tokens'],0);self.assertEqual(result['map_loss'],0)
        self.assertTrue(math.isfinite(result['loss']))

    def test_batchnorm_and_dropout_rejected(self):
        for module in (nn.BatchNorm1d(8),nn.Dropout(.1),nn.MultiheadAttention(8,2,dropout=.1)):
            model=Tiny().train();model.unsupported=module
            with self.assertRaises(ValueError):backward_effective_batch(model,sample_batch(5),1)

    def test_recompute_preserves_RNG_and_gradients(self):
        plain=nn.Sequential(nn.Linear(8,8),nn.Dropout(.4),nn.GELU(),nn.Linear(8,8))
        recomputed=copy.deepcopy(plain);x=torch.randn(3,8,requires_grad=True);y=x.detach().clone().requires_grad_(True)
        torch.manual_seed(22);plain(x).square().sum().backward()
        torch.manual_seed(22);recompute_block(recomputed,y).square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        for p,q in zip(plain.parameters(),recomputed.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_Adam_step_once_and_scheduler_epoch_only(self):
        model=Tiny().train();config=TrainConfig();optimizer=build_optimizer(model,config);scheduler=build_scheduler(optimizer,config)
        before=model.image.weight.detach().clone()
        for _ in range(2):
            result=optimizer_update(model,optimizer,sample_batch(32),1);self.assertEqual(result['optimizer_steps'],1)
        self.assertEqual(scheduler.last_epoch,0)
        self.assertFalse(torch.equal(before,model.image.weight))
        for value in optimizer.state.values():
            self.assertEqual(int(value['step']),2);self.assertEqual(value['exp_avg'].dtype,torch.float32)
            self.assertEqual(value['exp_avg_sq'].dtype,torch.float32)
        scheduler.step();self.assertEqual(scheduler.last_epoch,1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],2.8e-6)

    def test_all_60_epochs_schedule_and_original_config(self):
        _,config,memory=load_config(ROOT/'configs/train_fp32.json')
        old=json.loads((ROOT/'reports/baseline_config.json').read_text())
        from dataclasses import asdict
        self.assertEqual(json.dumps(asdict(config),sort_keys=True),json.dumps(old['train'],sort_keys=True))
        optimizer=build_optimizer(Tiny(),config);scheduler=build_scheduler(optimizer,config)
        lrs=[optimizer.param_groups[0]['lr']]
        for _ in range(60):optimizer.step();scheduler.step();lrs.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(lrs[0],1e-6);self.assertAlmostEqual(lrs[5],1e-5);self.assertEqual(lrs[60],0)
        self.assertTrue(all(a>=b for a,b in zip(lrs[5:],lrs[6:])))
        for size in (1,2,4):
            path=ROOT/'configs'/('train_fp32.json' if size==1 else 'train_fp32_micro%d.json'%size)
            _,_,m=load_config(path);m.validate(32);self.assertEqual(m.micro_batch_size,size)
        with self.assertRaises(ValueError):MemoryConfig(micro_batch_size=2,accumulation_steps=8).validate(32)

    def test_only_ten_plain_state_dict_files_for_60_epochs(self):
        model=Tiny()
        for epoch in range(1,61):
            result=save_epoch_weights(model,self.root,epoch)
            self.assertEqual(result is not None,epoch<=10)
        files=sorted(p.name for p in self.root.iterdir())
        self.assertEqual(files,['epoch_%03d.pth'%i for i in range(1,11)])
        for name in files:
            value=load_state_dict(self.root/name)
            self.assertEqual(set(value),set(model.state_dict()))
            self.assertTrue(all(torch.is_tensor(v) for v in value.values()))
            fresh=Tiny();fresh.load_state_dict(value,strict=True)
            for k,v in fresh.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
        with self.assertRaises(FileExistsError):save_epoch_weights(model,self.root,1)
        with self.assertRaises(ValueError):save_epoch_weights(model,self.root,0)
        with self.assertRaises(ValueError):save_epoch_weights(model,self.root,61)

    def test_engine_continues_training_and_validation_all_60_epochs(self):
        queries=tuple(attrs(i) for i in range(3))
        records=tuple(ImageRecord('train/%d.jpg'%i,self.root/('%d.jpg'%i),attrs(i),i%3) for i in range(5))
        split=UPARSplit('train',records,queries)
        sampler=QueryBalancedSampler(split,1);data=sample_batch(5)
        dataset=[{k:v[i] for k,v in data.items()} for i in range(5)]
        loader=DataLoader(dataset,batch_size=32,sampler=sampler,generator=torch.Generator().manual_seed(1))
        evaluated=[]
        def fake_evaluate(model,*args):
            evaluated.append(len(evaluated)+1)
            return {k:.4 for k in ('mADM','mAP','Rank-1','Rank-5','Rank-10','mINP')}
        run=self.root/'run'
        train(Tiny(),(loader,None,None),TrainConfig(num_workers=0),MemoryConfig(),torch.device('cpu'),run,
              evaluator=fake_evaluate)
        status=json.loads((run/'status.json').read_text());self.assertEqual(status['phase'],'complete')
        self.assertEqual(status['completed_epochs'],60);self.assertEqual(status['global_step'],60)
        self.assertEqual(status['scheduler_steps'],60);self.assertEqual(len(evaluated),60)
        self.assertEqual(sorted(p.name for p in run.glob('*.pth')),['epoch_%03d.pth'%i for i in range(1,11)])
        metrics=[json.loads(row) for row in (run/'metrics.jsonl').read_text().splitlines()]
        self.assertEqual([m['epoch'] for m in metrics],list(range(1,61)))
        self.assertTrue(all(m['weight'] is None for m in metrics[10:]))
        with (run/'metrics.csv').open() as stream:self.assertEqual(len(list(csv.DictReader(stream))),60)

    def test_metric_and_loss_math_match_original_AST(self):
        base=ROOT.parent/'IRRA-QueryBalanced'
        for name,functions in (('upar/metrics.py',('metrics_for_ranking','retrieval_metrics')),
                               ('model/objectives.py',('_sdm_float32','compute_map'))):
            old=ast.parse((base/name).read_text());new=ast.parse((ROOT/name).read_text())
            for function in functions:
                a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==function)
                b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==function)
                self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False))

    def test_chunked_retrieval_matches_original_full_features(self):
        model=Tiny().eval();data=sample_batch(9)
        query=[(data['attributes'][i],data['caption_ids'][i]) for i in range(9)]
        gallery=[(data['attributes'][i],data['images'][i]) for i in range(9)]
        with torch.no_grad():
            expected=retrieval_metrics(model.encode_text(data['caption_ids']),model.encode_image(data['images']),
                                       data['attributes'],data['attributes'])
        got=evaluate(model,DataLoader(gallery,batch_size=9),DataLoader(query,batch_size=9),32,2)
        self.assertEqual(got,expected)

    def test_sampler_frequencies_prompt_and_leak_guard(self):
        queries=tuple(attrs(i) for i in range(3))
        records=tuple(ImageRecord('PETA/train%d.jpg'%i,self.root/('train%d.jpg'%i),attrs(i),i%3) for i in range(8))
        validation=tuple(replace(r,relative_path=r.relative_path.replace('train','val'),image_path=self.root/('val%d.jpg'%i)) for i,r in enumerate(records))
        split=UPARSplit('train',records,queries);val=UPARSplit('val',validation,queries)
        sampler=QueryBalancedSampler(split,1)
        self.assertEqual(list(sampler),list(QueryBalancedSampler(split,1)))
        for record,w in zip(records,sampler.weights):self.assertAlmostEqual(float(w),sampler.counts[record.attributes]**-.5)
        self.assertEqual(audit_splits(split,val)['train_validation_id_overlap'],0)
        with self.assertRaises(ValueError):audit_splits(split,replace(val,images=(records[0],)+validation[1:]))
        tokenizer=PromptTokenizer()
        for row in queries:
            prompt=encode_query(row,tokenizer)
            self.assertEqual(len(prompt.token_ids),77);self.assertEqual(prompt.token_ids.count(49407),1)

    def test_positional_H_W_CLS_and_FP32_policy(self):
        original=torch.randn(257,1024);resized=resize_visual_position(original)
        self.assertEqual(tuple(resized.shape),(244,1024));self.assertEqual(resized.dtype,torch.float32)
        torch.testing.assert_close(original[0],resized[0],rtol=0,atol=0)
        model=Tiny().train()
        with patch('torch.autocast',side_effect=AssertionError('AMP forbidden')), \
             patch('torch.cuda.amp.autocast',side_effect=AssertionError('AMP forbidden')), \
             patch('torch.cuda.amp.GradScaler',side_effect=AssertionError('AMP forbidden')):
            backward_effective_batch(model,sample_batch(5),1)
        assert_model_fp32(model,gradients=True)
        forbidden=('autocast','GradScaler','half','bfloat16','convert_weights')
        for directory in ('model','upar'):
            for path in (ROOT/directory).glob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node,ast.Call):self.assertNotIn(getattr(node.func,'attr',getattr(node.func,'id','')),forbidden,str(path))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Core))
    value=dict(passed=result.wasSuccessful(),tests_run=result.testsRun,gpu_used=False,
               numerical_comparisons=COMPARISONS,errors=[str(t) for t,e in result.errors],failures=[str(t) for t,e in result.failures])
    (ROOT/'reports/unit_tests.json').write_text(json.dumps(value,indent=2)+'\n')
    sys.exit(0 if result.wasSuccessful() else 1)
