"""One extraction per validation; shared scores, CPU sweep, official metrics."""
import csv
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import metrics_for_ranking
from .local_scoring import fuse,stable_rank

METRICS=('mADM','mAP','Rank-1','Rank-5','Rank-10','mINP')
KST=timezone(timedelta(hours=9))
WORKER={}


def log(text):
    print(datetime.now(KST).isoformat(timespec='seconds')+' '+text,flush=True)


def _init_sweep(directory,queries,gallery,grid):
    WORKER.update(scores={k:np.load(str(Path(directory)/(k+'_scores.npy')),mmap_mode='r')
                          for k in ('global','nll','agreement')},queries=queries,gallery=gallery,grid=grid,
                  positions=np.arange(1,len(gallery)+1,dtype=np.float64))


def _chunk(start,end):
    result={(m,w):np.empty(end-start) for m,weights in WORKER['grid'].items() for w in weights}
    for j,qi in enumerate(range(start,end)):
        agreements=(WORKER['gallery']==WORKER['queries'][qi]).sum(1)
        relevant=agreements==40
        dom=agreements.astype(np.float64)/40
        denominator=dom.max()-dom.mean()
        ndom=np.ones_like(dom) if denominator==0 else np.maximum((dom-dom.mean())/denominator,0)
        g=WORKER['scores']['global'][qi:qi+1]
        for method,weights in WORKER['grid'].items():
            a=WORKER['scores'][method][qi:qi+1]
            rows=np.concatenate([fuse(g,a,w) for w in weights],axis=0) if weights else np.empty((0,len(dom)))
            rankings=stable_rank(rows)
            for k,w in enumerate(weights):
                ranking=rankings[k]
                degrees=np.cumsum(ndom[ranking])/WORKER['positions']
                result[(method,w)][j]=degrees[relevant[ranking]].mean()
    return start,end,result


def sweep(directory,queries,gallery,grid,progress=None):
    output={(m,w):np.empty(len(queries)) for m,weights in grid.items() for w in weights}
    with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('fork'),
                             initializer=_init_sweep,initargs=(str(directory),queries,gallery,grid)) as pool:
        futures=[pool.submit(_chunk,s,min(s+100,len(queries))) for s in range(0,len(queries),100)]
        completed=0
        for future in as_completed(futures):
            start,end,values=future.result()
            for key,a in values.items():output[key][start:end]=a
            completed+=end-start
            if progress is not None:progress(completed,len(queries))
            if completed%800==0 or completed==len(queries):log('Fusion sweep %d/%d'%(completed,len(queries)))
    return output


def _mean(array,mask=None):
    a=array if mask is None else array[mask]
    return {name:float(a[:,k].mean()) for k,name in enumerate(METRICS)} if len(a) else None


@torch.inference_mode()
def evaluate_local(model,gallery_loader,query_loader,output,train_queries,weights=None,
                   micro_batch_size=2,progress=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    original_training=model.training;model.eval()
    device=next(model.parameters()).device
    try:
        qf,qa,gf,ga,gz=[],[],[],[],[]
        query_seen=0
        for attrs,tokens in query_loader:
            for start in range(0,len(tokens),micro_batch_size):
                qf.append(model.encode_text(tokens[start:start+micro_batch_size].to(device)).cpu())
            qa.append(attrs)
            query_seen+=len(tokens)
            if progress is not None:progress('query_features',query_seen,len(query_loader.dataset))
        for batch,(attrs,images) in enumerate(gallery_loader,1):
            for start in range(0,len(images),micro_batch_size):
                features,logits=model.encode_image_and_attributes(images[start:start+micro_batch_size].to(device))
                gf.append(features.cpu());gz.append(logits.cpu())
            ga.append(attrs)
            if progress is not None:progress('gallery_features',batch,len(gallery_loader))
            if batch==1 or batch%40==0 or batch==len(gallery_loader):
                log('Validation gallery %d/%d batches'%(batch,len(gallery_loader)))
        qf,qa,gf,ga,gz=map(torch.cat,(qf,qa,gf,ga,gz))
        if not all(torch.isfinite(t).all() for t in (qf,gf,gz)):
            raise ValueError('Nonfinite extracted validation feature/logit')
        seen=np.array([tuple(q) in train_queries for q in qa.tolist()],dtype=bool)
        torch.save({'query_features':qf,'query_attributes':qa,'gallery_features':gf,
                    'gallery_attributes':ga,'gallery_logits':gz,'seen':seen.tolist()},output/'features.pt')
        queries,gallery=F.normalize(qf.float(),dim=1),F.normalize(gf.float(),dim=1)
        score={name:np.lib.format.open_memmap(str(output/(name+'_scores.npy')),mode='w+',
               dtype=np.float32 if name=='global' else np.float64,shape=(len(qa),len(ga)))
               for name in ('global','nll','agreement')}
        z=gz.double();p=torch.sigmoid(z);neg=F.logsigmoid(-z).sum(1)
        for start in range(0,len(qa),32):
            q=qa[start:start+32].double()
            score['global'][start:start+32]=(queries[start:start+32]@gallery.t()).numpy()
            score['nll'][start:start+32]=(neg[None,:]+q@z.t()).numpy()
            score['agreement'][start:start+32]=(((1-p).sum(1)[None,:]+q@(2*p-1).t())/40).numpy()
        for a in score.values():
            a.flush()
            if not np.isfinite(a).all():raise ValueError('Nonfinite cached score')
        grid=weights or [round(i/10,2) for i in range(11)]
        if 0 not in grid or 1 not in grid:raise ValueError('Fusion grid must include 0 and 1')
        values=sweep(output,qa.numpy(),ga.numpy(),{'nll':grid,'agreement':grid},
                     None if progress is None else lambda done,total:progress('coarse_sweep',done,total))
        coarse={m:max(grid,key=lambda w:(values[(m,w)].mean(),-w)) for m in ('nll','agreement')}
        fine={m:[round(i/100,2) for i in range(max(0,int(round(w*100))-10),min(100,int(round(w*100))+10)+1,2)
                 if round(i/100,2) not in grid] for m,w in coarse.items()}
        values.update(sweep(output,qa.numpy(),ga.numpy(),fine,
                            None if progress is None else lambda done,total:progress('fine_sweep',done,total)))
        best={m:max([w for mm,w in values if mm==m],key=lambda w:(values[(m,w)].mean(),-w)) for m in ('nll','agreement')}
        with (output/'fusion_grid.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['attribute_score','attribute_weight','mADM_percent','best'])
            for (m,w),a in sorted(values.items()):writer.writerow([m,w,a.mean()*100,int(best[m]==w)])
        names={'Global-only':('global',None),'NLL-only':('nll',None),'Agreement-only':('agreement',None),
               'Global+NLL':('nll',best['nll']),'Global+Agreement':('agreement',best['agreement']),
               'Global+NLL-fixed0.40':('nll',0.40)}
        official={name:np.empty((len(qa),len(METRICS))) for name in names}
        maximum_error=0.0
        for qi in range(len(qa)):
            for name,(m,w) in names.items():
                row=score[m][qi] if w is None else fuse(score['global'][qi:qi+1],score[m][qi:qi+1],w)[0]
                metrics=metrics_for_ranking(stable_rank(row),qa[qi],ga)
                official[name][qi]=[metrics[k] for k in METRICS]
                if w is not None and (m,w) in values:
                    error=abs(metrics['mADM']-values[(m,w)][qi]);maximum_error=max(maximum_error,error)
                    if error>1e-12:raise AssertionError('Sweep/official mADM differs')
            if qi%500==0 or qi+1==len(qa):log('Official metrics %d/%d'%(qi+1,len(qa)))
            if progress is not None and (qi%500==0 or qi+1==len(qa)):
                progress('official_metrics',qi+1,len(qa))
        np.savez(output/'per_query_metrics.npz',**official)
        result={'completed_kst':datetime.now(KST).isoformat(),'queries':len(qa),'gallery':len(ga),
                'seen_queries':int(seen.sum()),'unseen_queries':int((~seen).sum()),'best_weights':best,
                'methods':{name:_mean(a) for name,a in official.items()},
                'seen':{name:_mean(a,seen) for name,a in official.items()},
                'unseen':{name:_mean(a,~seen) for name,a in official.items()},
                'sweep_official_max_abs_error':maximum_error,
                'attribute_BCE':F.binary_cross_entropy_with_logits(z,ga.double()).item(),
                'vs_same_global':{name:{'mADM_delta_pp':float((a[:,0]-official['Global-only'][:,0]).mean()*100),
                    'improved_queries':int((a[:,0]>official['Global-only'][:,0]+1e-11).sum()),
                    'worse_queries':int((a[:,0]<official['Global-only'][:,0]-1e-11).sum())}
                                  for name,a in official.items()}}
        (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        log('Validation result '+json.dumps(result['methods']))
        # Only temporary scores created inside this brand-new evaluation directory.
        # Retain all features/logits, per-query metrics and grids for reproducibility.
        score.clear()
        for name in ('global','nll','agreement'):(output/(name+'_scores.npy')).unlink()
        return result
    finally:
        model.train(original_training)
