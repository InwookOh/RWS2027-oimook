"""Strict foundation loading; only original OpenAI ViT-L/14 is accepted."""
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from model.clip_model import CLIP

OFFICIAL_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
OFFICIAL_URL = 'https://openaipublic.azureedge.net/clip/models/' + OFFICIAL_SHA256 + '/ViT-L-14.pt'
CLIP_CONFIG = dict(embed_dim=768, image_resolution=(378, 126), vision_layers=24,
                   vision_width=1024, vision_patch_size=14, context_length=77,
                   vocab_size=49408, transformer_width=768, transformer_heads=12,
                   transformer_layers=12, stride_size=14)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(data)
    return h.hexdigest()


def resize_visual_position(position):
    if position.shape != (257, 1024) or position.dtype != torch.float32:
        raise ValueError('Require official ViT-L/14 position embedding in FP32')
    grid = position[1:].reshape(1, 16, 16, 1024).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(27, 9), mode='bilinear', align_corners=False)
    return torch.cat((position[:1], grid.permute(0, 2, 3, 1).reshape(243, 1024)), dim=0)


def load_pretrained(path):
    path = Path(path).resolve()
    digest = sha256(path)
    if digest != OFFICIAL_SHA256:
        raise ValueError('Official OpenAI ViT-L/14 hash mismatch')
    archive = torch.jit.load(str(path), map_location='cpu')
    raw = archive.state_dict()
    expected = {'visual.conv1.weight': (1024, 3, 14, 14),
                'visual.positional_embedding': (257, 1024), 'visual.proj': (1024, 768),
                'token_embedding.weight': (49408, 768), 'positional_embedding': (77, 768),
                'text_projection': (768, 768), 'ln_final.weight': (768,)}
    for name, shape in expected.items():
        if name not in raw or tuple(raw[name].shape) != shape: raise ValueError('Foundation shape: ' + name)
    for prefix, blocks in (('visual.transformer.resblocks.', 24), ('transformer.resblocks.', 12)):
        actual = {k[len(prefix):].split('.')[0] for k in raw if k.startswith(prefix)}
        if actual != set(map(str, range(blocks))): raise ValueError('Invalid transformer blocks')
    model = CLIP(**CLIP_CONFIG).float()
    keys = set(model.state_dict())
    missing = keys - set(raw)
    extras = set(raw) - keys
    if missing or extras - {'logit_scale', 'input_resolution', 'context_length', 'vocab_size'}:
        raise ValueError('Foundation keys missing=%s unexpected=%s' % (missing, extras))
    values = {key: raw[key].float() for key in keys}
    values['visual.positional_embedding'] = resize_visual_position(values['visual.positional_embedding'])
    model.load_state_dict(values, strict=True)
    for key, value in model.state_dict().items():
        if value.dtype != torch.float32 or not torch.equal(value, values[key]):
            raise ValueError('Foundation value/dtype mismatch: ' + key)
    report = dict(path=str(path), sha256=digest, verified_backbone_keys=len(keys), missing_keys=[],
                  unexpected_keys=[], shape_mismatches=[], ignored_scalars=sorted(extras),
                  source_grid=[16, 16], target_grid=[27, 9], target_position_shape=[244, 1024],
                  position_resize='FP32 bilinear align_corners=False; CLS preserved', clip_config=CLIP_CONFIG)
    return model, dict(CLIP_CONFIG), report
