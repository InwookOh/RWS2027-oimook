"""Launch one tested, fresh 60-Epoch run in isolated tmux; stop nothing."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=root/'configs/train_fp32_micro4.json')
    p.add_argument('--gpu-uuid')
    p.add_argument('--data-root',type=Path,default=root.parent/'data')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    config=json.loads(args.config.read_text());micro=config['memory']['micro_batch_size']
    unit=json.loads((root/'reports/unit_tests.json').read_text())
    smoke=json.loads((root/'reports'/('gpu_smoke_micro%d.json'%micro)).read_text())
    if not unit['passed'] or not smoke['passed'] or smoke['oom']:
        raise RuntimeError('Unit and actual GPU checks must pass before full training')
    if (smoke['effective_batch_size']!=32 or smoke['precision']!='FP32'
            or config['train']['num_epoch']!=60 or config['train']['batch_size']!=32
            or micro!=smoke['micro_batch_size']
            or config['memory']['accumulation_steps']!=smoke['accumulation_steps']):
        raise ValueError('Training configuration does not match validated settings')
    selected=args.gpu_uuid or smoke['gpu']['uuid']
    if selected!=smoke['gpu']['uuid']:raise ValueError('Test the requested GPU first')
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    name='full_'+now.strftime('%Y%m%d_%H%M%S_KST')+'_'+uuid.uuid4().hex[:6]
    run=root/'experiments'/name
    session='qb-vitl14-'+now.strftime('%Y%m%d-%H%M%S')
    tmux_socket='/tmp/irra-qb-vitl14-single.sock'
    command=[sys.executable,'-B','-u',str(root/'train_upar.py'),'--output',str(run),
             '--config',str(args.config.resolve()),'--data-root',str(args.data_root.resolve()),
             '--pretrained',str(root/'weights/ViT-L-14.pt'),'--gpu-uuid',selected,'--device','cuda']
    env=dict(CUDA_VISIBLE_DEVICES=selected,CUDA_DEVICE_ORDER='PCI_BUS_ID',NVIDIA_TF32_OVERRIDE='0',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',WORLD_SIZE='1',PYTHONPATH=str(root))
    plan=dict(run=str(run),session=session,tmux_socket=tmux_socket,command=command,environment=env,
              gpu=selected,config=str(args.config.resolve()),checkpoint_policy='plain state_dict epochs1..10 only',
              started_at_kst=now.isoformat(),dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(plan,indent=2));return
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                  '--format=csv,noheader,nounits'],universal_newlines=True)
    if any(row.split(',')[0].strip()==selected for row in apps.splitlines()):raise RuntimeError('GPU has an active process')
    gpu=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid,name,memory.total,memory.used',
                                 '--format=csv,noheader,nounits'],universal_newlines=True).strip()
    if int(gpu.split(',')[-1].strip())>256:raise RuntimeError('GPU memory is occupied')
    if run.exists():raise FileExistsError(run)
    existing=subprocess.run(['tmux','-S',tmux_socket,'has-session','-t',session],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if existing.returncode==0:raise FileExistsError('Session already exists')
    launch_dir=root/'reports/launches';launch_dir.mkdir(exist_ok=True)
    script=launch_dir/(name+'.sh');console=root/'reports'/(name+'.console.log')
    lock='/tmp/irra-qb-vitl14-%d-%s.lock'%(os.getuid(),selected)
    wrapped=['env']+[k+'='+v for k,v in env.items()]+['flock','-n',lock]+command
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(root))+'\nexec '+
                      ' '.join(shlex.quote(v) for v in wrapped)+' >> '+shlex.quote(str(console))+' 2>&1\n')
    plan.update(console_log=str(console),launch_script=str(script),gpu_preflight=gpu)
    (launch_dir/(name+'.json')).write_text(json.dumps(plan,indent=2)+'\n')
    subprocess.run(['tmux','-S',tmux_socket,'new-session','-d','-s',session,'-c',str(root),
                    'bash '+shlex.quote(str(script))],check=True)
    print(json.dumps(plan,indent=2))
    print('Attach: tmux -S %s attach-session -t %s'%(tmux_socket,session))


if __name__=='__main__':main()
