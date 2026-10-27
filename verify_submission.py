"""Run an extracted submission in an isolated Python process with network blocked.

This checks the API using public validation images, not hidden test data. It does
not substitute for testing in the organizer's CUDA 12.1 container/time limit.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from upar.attributes import ATTRIBUTES
from upar.data import load_split


RUNNER = r'''
import importlib.util, json, os, socket, sys, time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
def deny_network(*args, **kwargs):
    raise RuntimeError("Network access is forbidden during submission verification")
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
root = Path(sys.argv[1])
sample = json.loads(Path(sys.argv[2]).read_text())
spec = importlib.util.spec_from_file_location('submission_run', root / 'run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
started = time.monotonic()
if hasattr(module, 'load_model'):
    module.load_model()
result = module.rank_gallery(sample)
assert isinstance(result, dict) and len(result) == 1, 'Return exactly one key'
key = next(iter(result))
assert key in ('distances', 'similarities', 'ranking'), 'Invalid return key'
value = result[key]
if torch.is_tensor(value):
    value = value.detach().cpu().numpy()
value = np.asarray(value)
q, g = len(sample['queries']), len(sample['gallery'])
assert value.shape == (q, g), f'Wrong output shape: {value.shape}'
assert np.isfinite(value).all(), 'NaN/Inf in output'
if key == 'ranking':
    assert np.issubdtype(value.dtype, np.integer), 'Ranking must contain integers'
    assert np.all(np.sort(value, axis=1) == np.arange(g)), 'Invalid permutation'
np.save(root / 'result.npy', value)
print(f'Offline API OK: key={key}, shape={value.shape}, seconds={time.monotonic()-started:.2f}')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--max-images', type=int, default=32)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    names, _, queries, ids = load_split(args.data_dir, 'val', args.max_images)
    root = Path(args.data_dir).resolve()
    sample = {
        'attribute_names': list(ATTRIBUTES), 'queries': queries[np.unique(ids)].astype(int).tolist(),
        'gallery': [{'index': i, 'image': name, 'image_path': str(root / name)} for i, name in enumerate(names)],
    }
    with tempfile.TemporaryDirectory(prefix='upar-submission-') as folder:
        folder = Path(folder)
        with zipfile.ZipFile(args.zip) as archive:
            if not {'run.py', 'metadata.yaml'}.issubset(archive.namelist()):
                raise ValueError('run.py and metadata.yaml must be at archive root')
            for name in archive.namelist():
                if Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('Unsafe archive path')
            archive.extractall(folder)
        sample_path = folder / 'sample.json'; sample_path.write_text(json.dumps(sample))
        environment = os.environ.copy()
        if args.device == 'cpu':
            environment['CUDA_VISIBLE_DEVICES'] = ''
        subprocess.run([sys.executable, '-I', '-c', RUNNER, str(folder), str(sample_path)],
                       cwd=folder, env=environment, check=True)


if __name__ == '__main__':
    main()
