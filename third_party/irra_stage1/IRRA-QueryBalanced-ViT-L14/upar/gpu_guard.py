"""Choose only the visible, idle device by UUID; never terminate a process."""
import os
from pathlib import Path
import re
import subprocess


def configure_gpu(requested_uuid=None, require_idle=True):
    infos = [p.read_text() for p in Path('/proc/driver/nvidia/gpus').glob('*/information')]
    if not infos: raise RuntimeError('No visible GPU information')
    possible = [(re.search(r'GPU UUID:\s+(\S+)', info).group(1),
                 int(re.search(r'Device Minor:\s+(\d+)', info).group(1))) for info in infos]
    if requested_uuid is None:
        if len(possible) != 1: raise RuntimeError('Specify an exact GPU UUID')
        requested_uuid = possible[0][0]
    matches = [item for item in possible if item[0] == requested_uuid]
    if len(matches) != 1: raise ValueError('GPU is outside visible container devices')
    uuid, minor = matches[0]
    if not Path('/dev/nvidia%d' % minor).exists(): raise RuntimeError('Device node unavailable')
    rows = subprocess.check_output(['nvidia-smi', '-i', uuid,
                                   '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'],
                                  universal_newlines=True).strip()
    fields = [x.strip() for x in rows.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                        '--format=csv,noheader,nounits'], universal_newlines=True).strip()
    active = [row for row in processes.splitlines() if row.split(',')[0].strip() == uuid]
    if require_idle and (active or int(fields[3]) > 256):
        raise RuntimeError('GPU is occupied; no process will be stopped: %s %s' % (rows, active))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    import torch
    if torch.cuda.is_initialized(): raise RuntimeError('Select GPU before CUDA initialization')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one selected CUDA GPU is required')
    torch.cuda.set_device(0)
    return dict(uuid=uuid, physical_minor=minor, cuda_device='cuda:0', name=fields[1],
                total_memory_mib=int(fields[2]), prelaunch_memory_mib=int(fields[3]), other_compute_processes=active)
