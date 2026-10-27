"""ViT-L/14 QueryBalanced model with only the added local head trainable."""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from model.clip_model import CLIP
from .config import ModelConfig
from .local_branch import LocalAttributeBranch
from .model import UPARModel


ARCHITECTURE = 'qb-vit-l14-frozen-multiscale-local-8-16-24-v1'
NEW_PREFIXES = ('multiscale.', 'local_branch.')


@dataclass
class FrozenLocalConfig:
    attr_loss_weight: float = 0.5
    local_heads: int = 8
    fusion_weight: float = 0.0
    attribute_score: str = 'nll'
    multiscale_blocks: tuple = (8, 16, 24)
    initial_scale_weights: tuple = (0.2, 0.2, 0.6)


class WeightedPatchFusion(nn.Module):
    """Reference multiscale fusion: LN+projection at 8/16, projected block 24."""
    def __init__(self, weights=(0.2, 0.2, 0.6)):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape != (3,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('Three positive finite initial scale weights are required')
        if abs(float(weights.sum()) - 1.0) > 1e-6:
            raise ValueError('Initial scale weights must sum to one')
        self.norms = nn.ModuleDict({str(block): nn.LayerNorm(1024) for block in (8, 16)})
        self.projections = nn.ModuleDict({str(block): nn.Linear(1024, 768) for block in (8, 16)})
        self.fusion_logits = nn.Parameter(weights.log())

    def aligned(self, intermediate, final):
        if set(intermediate) != {8, 16} or final.ndim != 3 or final.shape[1:] != (244, 768):
            raise ValueError('Require blocks 8/16 and final CLIP tokens [B,244,768]')
        features = []
        for block in (8, 16):
            raw = intermediate[block]
            if raw.shape != (len(final), 243, 1024):
                raise ValueError('Expected intermediate ViT-L/14 patches [B,243,1024]')
            features.append(self.projections[str(block)](self.norms[str(block)](raw)))
        features.append(final[:, 1:])
        return features

    def forward(self, intermediate, final):
        features = self.aligned(intermediate, final)
        weights = self.fusion_logits.softmax(0)
        return sum(weights[i] * feature for i, feature in enumerate(features))


class FrozenMultiScaleLocalModel(UPARModel):
    """Retains every original QB state key; only ``multiscale`` and local head train."""
    def __init__(self, clip_model=None, clip_config=None, config=None):
        from .pretrained import CLIP_CONFIG
        self.local_config = config or FrozenLocalConfig()
        cfg = self.local_config
        if tuple(cfg.multiscale_blocks) != (8, 16, 24):
            raise ValueError('This experiment fixes transformer blocks 8/16/24')
        if cfg.local_heads != 8 or cfg.attr_loss_weight != 0.5 or cfg.attribute_score != 'nll':
            raise ValueError('Keep the reference local-head configuration (8 heads, 0.5 BCE, NLL)')
        if clip_config is not None and clip_config != CLIP_CONFIG:
            raise ValueError('Only the source ViT-L/14 CLIP configuration is allowed')
        clip_model = clip_model or CLIP(**CLIP_CONFIG)
        super().__init__(clip_model, config=ModelConfig(), with_map=True)
        # Match the reference construction order and isolate the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            self.local_branch = LocalAttributeBranch(dim=768, heads=cfg.local_heads, attributes=40)
        with torch.random.fork_rng(devices=[]):
            self.multiscale = WeightedPatchFusion(cfg.initial_scale_weights)
        self.configure_frozen_training()

    def configure_frozen_training(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.multiscale, self.local_branch):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.eval()
        return self

    def train(self, mode=True):
        # First set every module's mode, then restore every old QB component to eval.
        nn.Module.train(self, mode)
        if mode:
            for name, module in self.named_children():
                if name not in ('multiscale', 'local_branch'):
                    module.eval()
            self.multiscale.train(True)
            self.local_branch.train(True)
        return self

    def image_tokens_and_scales(self, images):
        if images.dtype != torch.float32:
            raise TypeError('Image input must remain FP32')
        # Backbone activations are constants; new projections/attention retain grad.
        with torch.no_grad():
            final, intermediate = self.base_model.visual(images, return_intermediate=True)
        if (final.shape[1:] != (244, 768) or set(intermediate) != {8, 16}
                or any(intermediate[k].shape[1:] != (243, 1024) for k in (8, 16))):
            raise RuntimeError('Unexpected ViT-L/14 multiscale tensor shape')
        return final, intermediate

    def local_from_tokens(self, final, intermediate):
        patches = self.multiscale(intermediate, final)
        return self.local_branch(patches)[0]

    def train_local_batch(self, images, attributes):
        if not self.training:
            raise RuntimeError('Set the added modules to training mode first')
        if attributes.shape != (len(images), 40) or not ((attributes == 0) | (attributes == 1)).all():
            raise ValueError('Require binary [B,40] attributes in canonical UPAR order')
        final, intermediate = self.image_tokens_and_scales(images)
        logits = self.local_from_tokens(final, intermediate)
        bce = F.binary_cross_entropy_with_logits(logits.float(), attributes.float())
        loss = bce * self.local_config.attr_loss_weight
        return {'loss': loss, 'attr_bce': bce, 'attribute_logits': logits,
                'global_features': final[:, 0].float()}

    def encode_image_and_attributes(self, images):
        final, intermediate = self.image_tokens_and_scales(images)
        return final[:, 0].float(), self.local_from_tokens(final, intermediate)

    def trainable_parameter_names(self):
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def old_qb_state_digest(self):
        """Digest all original QB parameters and buffers in sorted key order."""
        import hashlib
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if name.startswith(NEW_PREFIXES):
                continue
            value = tensor.detach().to('cpu').contiguous()
            digest.update(name.encode('utf-8') + b'\0')
            digest.update(str(value.dtype).encode('ascii') + b'\0')
            digest.update(repr(tuple(value.shape)).encode('ascii') + b'\0')
            digest.update(memoryview(value.numpy()).cast('B'))
        return digest.hexdigest()

