"""A teammate's E1 model (CLIP ViT-L/14 AttriVision + AttriQuery local head) as an ensemble member.

E1 scores 52 category states twice: globally (normalized CLIP image embedding against the
fixed state-prompt embeddings) and locally (12 category queries cross-attend to the final
patch tokens). Each score is turned into per-category log-probabilities at temperature T and
the two are averaged (alpha 0.5), as in its evaluator. Here the fused probability of each
attribute's own state is returned as a 40-attribute logit, so the member mixes with ours.
Plain PyTorch (no open_clip): the state-prompt embeddings are stored with the weights
(converted by scripts/e1_member.py).
"""
import torch
from torch import nn
import torch.nn.functional as F

from .vit import VIT_SPECS, CLIPVisionTransformer

NUM_STATES = 52
NUM_CATEGORIES = 12


class CategoryCrossAttention(nn.Module):
    """One learnable query per category over the patch tokens (E1's AttriQuery local head)."""

    def __init__(self, num_categories=NUM_CATEGORIES, visual_dim=1024, text_dim=768, num_heads=8):
        super().__init__()
        self.category_queries = nn.Parameter(torch.zeros(num_categories, visual_dim))
        self.attention = nn.MultiheadAttention(visual_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(visual_dim)
        self.ffn = nn.Sequential(nn.Linear(visual_dim, 4 * visual_dim), nn.GELU(), nn.Dropout(0.0),
                                 nn.Linear(4 * visual_dim, visual_dim))
        self.norm2 = nn.LayerNorm(visual_dim)
        self.output_projection = nn.Linear(visual_dim, text_dim)

    def forward(self, patch_tokens):
        queries = self.category_queries.unsqueeze(0).expand(patch_tokens.shape[0], -1, -1)
        attended, _ = self.attention(queries, patch_tokens, patch_tokens, need_weights=False)
        output = self.norm1(queries + attended)
        output = self.norm2(output + self.ffn(output))
        return F.normalize(self.output_projection(output.float()), dim=-1)


class E1Member(nn.Module):
    def __init__(self, image_size=(224, 224)):
        super().__init__()
        self.visual = CLIPVisionTransformer(**VIT_SPECS['clip_vit_l14'], image_size=tuple(image_size))
        del self.visual.head  # E1 scores states in CLIP's joint space instead
        self.proj = nn.Parameter(torch.zeros(1024, 768))
        self.local_head = CategoryCrossAttention()
        self.register_buffer('state_features', torch.zeros(NUM_STATES, 768))  # unit-norm prompt embeddings
        self.register_buffer('state_category', torch.zeros(NUM_STATES, dtype=torch.long))  # category of each state
        self.register_buffer('state_attribute', torch.full((NUM_STATES,), -1, dtype=torch.long))  # -1: no attribute
        self.register_buffer('temperature', torch.tensor(0.01))
        self.register_buffer('alpha', torch.tensor(0.5))
        self.head_prefix = 'local_head.'
        self._indices = None

    def _layout(self):
        """Category state indices and attribute columns, computed once (fixed buffers; avoids a
        GPU sync per batch)."""
        if self._indices is None or self._indices[0][0].device != self.state_category.device:
            groups = [(self.state_category == c).nonzero(as_tuple=True)[0] for c in range(NUM_CATEGORIES)]
            present = (self.state_attribute >= 0).nonzero(as_tuple=True)[0]
            self._indices = (groups, present, self.state_attribute[present])
        return self._indices

    def state_log_probabilities(self, x):
        tokens = self.visual.tokens(x)  # final transformer tokens, before ln_post (E1's hook point)
        image = self.visual.ln_post(tokens[:, 0]) @ self.proj
        local = self.local_head(tokens[:, 1:].float())  # [B, 12, 768]
        # Scores and softmaxes in float32, also under autocast (scores are divided by T = 0.01).
        with torch.autocast(device_type=x.device.type, enabled=False):
            image = F.normalize(image.float(), dim=-1)
            local = F.normalize(local.float(), dim=-1)
            states = F.normalize(self.state_features.float(), dim=-1)
            global_scores = (image @ states.T) / self.temperature.float()  # [B, 52]
            local_scores = torch.einsum('bcd,sd->bcs', local, states) / self.temperature.float()  # [B, 12, 52]
            return self._fuse(global_scores, local_scores)

    def _fuse(self, global_scores, local_scores):
        fused = torch.empty_like(global_scores)
        for category, index in enumerate(self._layout()[0]):
            global_log = torch.log_softmax(global_scores[:, index], dim=1)
            local_log = torch.log_softmax(local_scores[:, category, index], dim=1)
            alpha = self.alpha.float()
            fused[:, index] = (1 - alpha) * global_log + alpha * local_log
        return fused

    def forward(self, x):
        fused = self.state_log_probabilities(x)
        _, present, columns = self._layout()
        logp = torch.full((x.shape[0], 40), -30.0, device=x.device, dtype=torch.float32)
        logp[:, columns] = fused[:, present]
        logp = logp.clamp(max=-1e-6)
        return logp - torch.log1p(-logp.exp())
