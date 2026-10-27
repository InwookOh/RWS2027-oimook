"""Original QueryBalanced SDM+MAP, with the ViT-L/14 global and token dimensions."""
from dataclasses import asdict
from types import SimpleNamespace
import torch
from model.build import IRRA
from model.clip_model import CLIP, Transformer
from model.objectives import compute_attribute_sdm, compute_map
from .config import ModelConfig
from .masking import IGNORE_INDEX
from .pretrained import CLIP_CONFIG, load_pretrained
from .precision import assert_model_fp32


class UPARModel(IRRA):
    def __init__(self, encoder, config=None, with_map=True):
        self.config = config or ModelConfig()
        if asdict(self.config) != asdict(ModelConfig()):
            # JSON changes tuple geometry into a list; compare it canonically.
            import json
            if json.dumps(asdict(self.config), sort_keys=True) != json.dumps(asdict(ModelConfig()), sort_keys=True):
                raise ValueError('Preserve QueryBalanced model settings except ViT-L/14 geometry')
        args = SimpleNamespace(**asdict(self.config), loss_names='sdm+map' if with_map else 'sdm')
        super().__init__(args, 0, base_model=encoder, base_cfg=CLIP_CONFIG)
        self.clip_config = dict(CLIP_CONFIG)
        self.with_map = with_map
        del self.logit_scale
        self.register_buffer('logit_scale', torch.tensor(1 / self.config.temperature, dtype=torch.float32))
        self.float()
        assert_model_fp32(self)

    @classmethod
    def from_pretrained(cls, path):
        encoder, config, report = load_pretrained(path)
        model = cls(encoder)
        model.pretrained_report = report
        return model

    @classmethod
    def from_state_dict(cls, state, with_map=True):
        if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
            raise ValueError('Require a plain model state_dict, not a training checkpoint')
        if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
            raise TypeError('Saved weights must be FP32')
        model = cls(CLIP(**CLIP_CONFIG), with_map=with_map)
        if not with_map:
            # Original retrieval uses only the encoders and fixed temperature, not MAP.
            state = {k: v for k, v in state.items() if k.startswith('base_model.') or k == 'logit_scale'}
        model.load_state_dict(state, strict=True)
        assert_model_fp32(model)
        return model

    def gradient_checkpointing(self, enabled=True):
        for module in self.modules():
            if isinstance(module, Transformer): module.gradient_checkpointing = bool(enabled)

    def encode_text(self, tokens):
        eos = tokens.eq(49407)
        if not eos.sum(dim=1).eq(1).all(): raise ValueError('Invalid Prompt EOS')
        features = self.base_model.encode_text(tokens)
        return features[torch.arange(len(tokens), device=tokens.device), eos.long().argmax(dim=1)]

    def global_features(self, batch):
        if batch['images'].dtype != torch.float32: raise TypeError('Images must be FP32')
        image, text = self.base_model(batch['images'], batch['caption_ids'])
        eos = batch['caption_ids'].eq(49407)
        if not eos.sum(dim=1).eq(1).all(): raise ValueError('Invalid Prompt EOS')
        global_text = text[torch.arange(len(text), device=text.device), eos.long().argmax(dim=1)]
        return image[:, 0], global_text, image

    def map_result(self, batch, image_tokens):
        if not self.with_map: raise RuntimeError('MAP is disabled for retrieval only')
        masked = self.base_model.encode_text(batch['map_ids'])
        fused = self.cross_former(masked, image_tokens, image_tokens)
        selected = batch['map_labels'].ne(IGNORE_INDEX)
        scores = self.mlm_head(fused[selected])
        labels = batch['map_labels'][selected]
        if scores.dtype != torch.float32: raise TypeError('MAP logits must be FP32')
        loss = compute_map(scores, labels) * self.config.map_loss_weight
        correct = scores.argmax(-1).eq(labels).sum() if len(labels) else selected.sum() * 0
        return dict(map_loss=loss, map_tokens=selected.sum(), map_correct=correct)

    def forward(self, batch):
        if not self.training: raise RuntimeError('Use global encode methods for retrieval')
        image, text, tokens = self.global_features(batch)
        result = self.map_result(batch, tokens)
        result['sdm_loss'] = compute_attribute_sdm(image, text, batch['attributes'], self.logit_scale)
        return result
