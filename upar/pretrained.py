"""Pretrained backbone initializations, downloaded and checked against pinned sha256 digests
(training only; submissions store full weights). The 2027 rules allow foundation models/VLMs
that were not trained for PAR."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import urllib.request

import numpy as np
import torch

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

SOURCES = {
    # OpenAI CLIP ViT-L/14 fine-tuned at 336x336 (24x24 grid); same keys as clip_openai_l14.
    'clip_openai_l14_336': {
        'architecture': 'clip_vit_l14', 'format': 'openai_clip',
        'url': 'https://huggingface.co/timm/vit_large_patch14_clip_336.openai/resolve/main/open_clip_pytorch_model.bin',
        'sha256': 'c7a812f61be88b4148e6c910ea245178ff3663263d54680cdb99dd6bcaed9b32',
    },
    # MetaCLIP ViT-L/14 (2.5B pairs curated with OpenAI's recipe; QuickGELU per the hub's
    # open_clip_config.json, OpenAI keys and normalization; CC-BY-NC 4.0). A different pretraining run of the same architecture.
    'clip_metaclip_l14': {
        'architecture': 'clip_vit_l14', 'format': 'openai_clip',
        'url': 'https://huggingface.co/timm/vit_large_patch14_clip_224.metaclip_2pt5b/resolve/main/open_clip_pytorch_model.bin',
        'sha256': '8982bc381934bb39006c26e370d12da1f1eff5748762489f7b22e1a350975061',
    },
    # OpenCLIP ViT-L/14 trained on LAION-2B (s32B-b82K; MIT). Exact GELU; normalization mean = std = 0.5.
    # DINOv2 ViT-L/14 (LVD-142M, self-supervised, image-only; CC-BY-NC 4.0), timm safetensors.
    # ImageNet normalization; no text tower.
    'dinov2_l14': {
        'architecture': 'dinov2_vit_l14', 'format': 'timm_dinov2',
        'url': 'https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m/resolve/main/model.safetensors',
        'sha256': '0424a5d1b515278cba3c6640ccbeaacc41de59d3a93df0dd5e494285eea2b355',
    },
    # Perception Encoder PE-Core ViT-L/14 at 336 (Meta 2025, CLIP-style; Apache-2.0), timm
    # safetensors of the image tower. Normalization mean = std = 0.5.
    'pe_core_l14': {
        'architecture': 'pe_core_vit_l14', 'format': 'timm_pe',
        'url': 'https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb/resolve/main/model.safetensors',
        'sha256': '35edcdb68eb43329192fc1530c246191e8a72f9a3dae01e2c7cccf05338d4cac',
    },
}


def read_safetensors(path):
    """Float32 tensors from a .safetensors file without the safetensors package."""
    with open(path, 'rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        data = stream.read()
    header.pop('__metadata__', None)
    tensors = {}
    for name, info in header.items():
        if info['dtype'] != 'F32':
            raise ValueError(f'{name}: unsupported dtype {info["dtype"]}')
        start, end = info['data_offsets']
        tensors[name] = torch.from_numpy(np.frombuffer(data[start:end], dtype=np.float32).reshape(info['shape']).copy())
    return tensors


def fetch(name, cache_dir=None):
    source = SOURCES[name]
    cache_dir = Path(cache_dir or os.environ.get('UPAR_PRETRAINED_DIR', 'outputs/pretrained'))
    path = cache_dir / name / source['url'].rsplit('/', 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Per-process temporary file: concurrent runs may download at once; os.replace is atomic.
        temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        with urllib.request.urlopen(source['url']) as response, open(temporary, 'wb') as out:
            shutil.copyfileobj(response, out)
        os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source['sha256']:
        raise ValueError(f'{path}: sha256 {digest} does not match the pinned {name} weights')
    return path


def load_pretrained_backbone(model, name, architecture, cache_dir=None):
    if architecture != SOURCES[name]['architecture']:
        raise ValueError(f'{name} weights require {SOURCES[name]["architecture"]}, got {architecture}')
    path = fetch(name, cache_dir)
    if SOURCES[name]['format'] == 'openai_clip':
        from .vit import openai_visual_state
        weights = openai_visual_state(torch.load(path, map_location='cpu', weights_only=True), model.grid)
    elif SOURCES[name]['format'] == 'timm_dinov2':
        from .vit import timm_dinov2_state
        weights = timm_dinov2_state(read_safetensors(path), model.grid)
    else:
        from .vit import timm_pe_state
        weights = timm_pe_state(read_safetensors(path), model.grid)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    # Only the new 40-attribute classifier is left at its random initialization.
    prefixes = model.head_prefix if isinstance(model.head_prefix, tuple) else (model.head_prefix,)
    head = prefixes[0]
    missing = [k for k in missing if not k.startswith(prefixes[1:])] if len(prefixes) > 1 else missing
    if unexpected or sorted(missing) != [head + 'bias', head + 'weight']:
        raise ValueError(f'Pretrained mismatch: missing={missing}, unexpected={unexpected}')
    return model
