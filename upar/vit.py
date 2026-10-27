"""CLIP vision transformer in plain PyTorch (submission runtime has no timm/open_clip).

Parameter names follow OpenAI CLIP's `visual` module, so pretrained weights load with
only the `visual.` prefix removed and the positional embedding resized to the input grid.
"""
from collections import OrderedDict
import math

import torch
from torch import nn
import torch.nn.functional as F

VIT_SPECS = {
    'clip_vit_l14': {'width': 1024, 'layers': 24, 'heads': 16, 'patch': 14},
    # DINOv2 ViT-L/14 (self-supervised, image-only): pre-norm blocks with LayerScale, exact GELU,
    # LayerNorm eps 1e-6, no pre-norm before the blocks; timm parameter names (DINOv2VisionTransformer).
    'dinov2_vit_l14': {'width': 1024, 'layers': 24, 'heads': 16, 'patch': 14, 'family': 'dinov2'},
    # Perception Encoder PE-Core ViT-L/14 image tower (Meta, CLIP-style contrastive): class token,
    # absolute positional embedding plus 2D RoPE on the patch tokens, pre-norm before the blocks,
    # LayerNorm eps 1e-5, 8-head attention pooling (PEVisionTransformer).
    'pe_core_vit_l14': {'width': 1024, 'layers': 24, 'heads': 16, 'patch': 14, 'family': 'pe'},
    # A two-block CLIP ViT for the unit tests only (fast checkpoints for soup and inference tests).
    'clip_vit_tiny': {'width': 32, 'layers': 2, 'heads': 2, 'patch': 16},
}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


def activation_layer(name):
    if name == 'quick_gelu':
        return QuickGELU()
    if name == 'gelu':
        return nn.GELU()
    raise ValueError(f'Unknown activation: {name}')


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(x.transpose(1, 2).reshape(batch, tokens, width))


class ResidualBlock(nn.Module):
    def __init__(self, width, heads, activation='quick_gelu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict(
            c_fc=nn.Linear(width, 4 * width), gelu=activation_layer(activation), c_proj=nn.Linear(4 * width, width)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, activation='quick_gelu'):
        super().__init__()
        self.resblocks = nn.ModuleList(ResidualBlock(width, heads, activation) for _ in range(layers))

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x


# Attribute indices of the 12 UPAR categories (age, gender, hair length, upper length,
# upper colour, lower length, lower colour, lower type, backpack, bag, glasses, hat).
CATEGORIES = ((0, 1, 2), (3,), (4, 5, 6), (7,), tuple(range(8, 20)), (20,), tuple(range(21, 33)),
              (33, 34), (35,), (36,), (37, 38), (39,))


class CategoryQueryHead(nn.Module):
    """One learnable query per attribute category attends over the patch tokens (as in
    Query2Label); each category's attended feature gives the logits of its attributes. The
    output layers start at zero, so a new head leaves the model's predictions unchanged."""

    def __init__(self, width, heads, num_classes=40, dropout=0.0):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(len(CATEGORIES), width) * width ** -0.5)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)
        self.out = nn.ModuleList(nn.Linear(width, len(c)) for c in CATEGORIES)
        for layer in self.out:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        self.register_buffer('index', torch.tensor([i for c in CATEGORIES for i in c]), persistent=False)
        self.num_classes = num_classes

    def forward(self, patches):
        queries = self.queries.to(patches.dtype).expand(patches.shape[0], -1, -1)
        attended, _ = self.attn(queries, patches, patches, need_weights=False)
        z = self.drop(self.norm(queries + attended))
        logits = torch.cat([layer(z[:, c]) for c, layer in enumerate(self.out)], dim=1)
        return logits.new_zeros(logits.shape[0], self.num_classes).index_copy(1, self.index, logits)


class CLIPVisionTransformer(nn.Module):
    """CLIP image encoder with the class token feeding a new attribute head."""

    head_prefix = 'head.'

    def __init__(self, width, layers, heads, patch, image_size, num_classes=40, dropout=0.0, query_head=False,
                 activation='quick_gelu'):
        super().__init__()
        height, width_px = image_size
        if height % patch or width_px % patch:
            raise ValueError(f'Input size {image_size} must be divisible by patch size {patch}')
        self.grid = (height // patch, width_px // patch)
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width) * width ** -0.5)
        self.positional_embedding = nn.Parameter(torch.randn(1 + self.grid[0] * self.grid[1], width) * width ** -0.5)
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, activation)
        self.ln_post = nn.LayerNorm(width)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)
        # Optional category-query head; its parameters live under `head_query.` so linear
        # probing (`head_prefix`) trains them together with `head.`.
        self.head_query = CategoryQueryHead(width, heads, num_classes, dropout) if query_head else None
        if query_head:
            self.head_prefix = ('head.', 'head_query.')

    def tokens(self, x):
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        return self.transformer(self.ln_pre(x))

    def features(self, x):
        return self.ln_post(self.tokens(x)[:, 0])

    def forward(self, x):
        if self.head_query is None:
            return self.head(self.head_drop(self.features(x)))
        x = self.ln_post(self.tokens(x))
        return self.head(self.head_drop(x[:, 0])) + self.head_query(x[:, 1:])



class DINOv2Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, width))


class LayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.gamma


class DINOv2Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = DINOv2Attention(width, heads)
        self.ls1 = LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(OrderedDict(fc1=nn.Linear(width, 4 * width), act=nn.GELU(),
                                             fc2=nn.Linear(4 * width, width)))
        self.ls2 = LayerScale(width)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DINOv2VisionTransformer(nn.Module):
    """DINOv2 image encoder (timm names) with the class token feeding a new attribute head."""

    head_prefix = 'head.'

    def __init__(self, width, layers, heads, patch, image_size, num_classes=40, dropout=0.0, query_head=False,
                 family='dinov2'):
        super().__init__()
        height, width_px = image_size
        if height % patch or width_px % patch:
            raise ValueError(f'Input size {image_size} must be divisible by patch size {patch}')
        self.grid = (height // patch, width_px // patch)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid[0] * self.grid[1], width))
        self.blocks = nn.ModuleList(DINOv2Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)
        # Optional category-query head on the patch tokens, added to the CLS head as in the CLIP
        # model; its parameters train with the head in the linear-probe epoch.
        self.head_query = CategoryQueryHead(width, heads, num_classes, dropout) if query_head else None
        if query_head:
            self.head_prefix = ('head.', 'head_query.')

    def tokens(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.to(x.dtype).expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed.to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def features(self, x):
        return self.tokens(x)[:, 0]

    def forward(self, x):
        if self.head_query is not None:
            x = self.tokens(x)
            return self.head(self.head_drop(x[:, 0])) + self.head_query(x[:, 1:])
        return self.head(self.head_drop(self.features(x)))


def rotate_interleaved(x):
    # [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
    return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)


class RopeAttention(nn.Module):
    """Attention with RoPE (interleaved layout) on the non-prefix tokens (PE)."""

    def __init__(self, width, heads, prefix_tokens, qkv_bias=False):
        super().__init__()
        self.heads = heads
        self.prefix_tokens = prefix_tokens
        self.rotate = rotate_interleaved
        self.qkv = nn.Linear(width, 3 * width, bias=qkv_bias)
        self.proj = nn.Linear(width, width)

    def forward(self, x, sin, cos):
        batch, tokens, width = x.shape
        q, k, v = self.qkv(x).view(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        n = self.prefix_tokens

        def rope(t):  # the class token keeps no position
            patches = t[:, :, n:]
            return torch.cat([t[:, :, :n], (patches * cos + self.rotate(patches) * sin).type_as(v)], dim=2)

        x = F.scaled_dot_product_attention(rope(q), rope(k), v)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, width))


class SigLIPBlock(nn.Module):
    """timm ViT block without LayerScale (SigLIP)."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = DINOv2Attention(width, heads)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(OrderedDict(fc1=nn.Linear(width, 4 * width), act=nn.GELU(),
                                             fc2=nn.Linear(4 * width, width)))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AttentionPool(nn.Module):
    """Multihead attention pooling of SigLIP and PE (timm AttentionPoolLatent, one latent query)."""

    def __init__(self, width, heads, eps=1e-6):
        super().__init__()
        self.heads = heads
        self.latent = nn.Parameter(torch.zeros(1, 1, width))
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.proj = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width, eps=eps)
        self.mlp = nn.Sequential(OrderedDict(fc1=nn.Linear(width, 4 * width), act=nn.GELU(),
                                             fc2=nn.Linear(4 * width, width)))

    def forward(self, x):
        batch, tokens, width = x.shape
        q = self.q(self.latent.to(x.dtype)).view(1, 1, self.heads, -1).transpose(1, 2).expand(batch, -1, -1, -1)
        k, v = self.kv(x).view(batch, tokens, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        x = self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, 1, width))
        return (x + self.mlp(self.norm(x)))[:, 0]


class SigLIPVisionTransformer(nn.Module):
    """SigLIP-style image tower (timm names), the base of PEVisionTransformer: the
    attention-pooled feature feeds the attribute head and the optional query head reads all
    (patch) tokens."""

    def __init__(self, width, layers, heads, patch, image_size, num_classes=40, dropout=0.0, query_head=False,
                 family='siglip'):
        super().__init__()
        height, width_px = image_size
        if height % patch or width_px % patch:
            raise ValueError(f'Input size {image_size} must be divisible by patch size {patch}')
        self.grid = (height // patch, width_px // patch)
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], width))
        self.blocks = nn.ModuleList(SigLIPBlock(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.attn_pool = AttentionPool(width, heads)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)
        self.head_query = CategoryQueryHead(width, heads, num_classes, dropout) if query_head else None
        self.head_prefix = ('head.', 'head_query.') if query_head else 'head.'

    def tokens(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2) + self.pos_embed.to(x.dtype)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def features(self, x):
        return self.attn_pool(self.tokens(x))

    def forward(self, x):
        if self.head_query is not None:
            x = self.tokens(x)
            return self.head(self.head_drop(self.attn_pool(x))) + self.head_query(x)
        return self.head(self.head_drop(self.features(x)))


def pe_rope(grid, head_dim, ref_grid=(24, 24), temperature=10000.0):
    """PE's 2D RoPE (timm RotaryEmbeddingCat with ref_feat_shape, grid offset 1, 'xy' indexing,
    interleaved layout): positions 1..n scaled to the pretraining grid. Returns sin, cos [H*W, head_dim]."""
    height, width = grid
    bands = 1.0 / temperature ** (torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 4))
    xs = (torch.arange(width, dtype=torch.float32) + 1) / width * ref_grid[1]
    ys = (torch.arange(height, dtype=torch.float32) + 1) / height * ref_grid[0]
    coords = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), dim=-1)  # [H, W, 2] as (x, y)
    angles = (coords[..., None] * bands).reshape(height * width, -1).repeat_interleave(2, dim=-1)
    return angles.sin(), angles.cos()


class PEBlock(nn.Module):
    """timm Eva block without LayerScale (PE)."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-5)
        self.attn = RopeAttention(width, heads, 1, qkv_bias=True)
        self.norm2 = nn.LayerNorm(width, eps=1e-5)
        self.mlp = nn.Sequential(OrderedDict(fc1=nn.Linear(width, 4 * width), act=nn.GELU(),
                                             fc2=nn.Linear(4 * width, width)))

    def forward(self, x, sin, cos):
        x = x + self.attn(self.norm1(x), sin, cos)
        return x + self.mlp(self.norm2(x))


class PEVisionTransformer(SigLIPVisionTransformer):
    """PE-Core image tower (timm names): class token, absolute positional embedding and RoPE;
    the attention-pooled feature (before PE's text-space projection) feeds the attribute head."""

    def __init__(self, width, layers, heads, patch, image_size, num_classes=40, dropout=0.0, query_head=False,
                 family='pe'):
        super().__init__(width, layers, heads, patch, image_size, num_classes, dropout, query_head, family)
        self.patch_embed.proj = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.grid[0] * self.grid[1], width))
        self.norm_pre = nn.LayerNorm(width, eps=1e-5)
        self.blocks = nn.ModuleList(PEBlock(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.attn_pool = AttentionPool(width, 8, eps=1e-5)
        sin, cos = pe_rope(self.grid, width // heads)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.register_buffer('rope_cos', cos, persistent=False)

    def tokens(self, x):
        x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.to(x.dtype).expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed.to(x.dtype)
        x = self.norm_pre(x)
        for block in self.blocks:
            x = block(x, self.rope_sin, self.rope_cos)
        return self.norm(x)

    def forward(self, x):
        if self.head_query is not None:
            x = self.tokens(x)
            return self.head(self.head_drop(self.attn_pool(x))) + self.head_query(x[:, 1:])
        return super().forward(x)


def timm_pe_state(state, grid):
    """timm PE checkpoint -> PEVisionTransformer keys: drops PE's text-space projection (`head.`)
    and resizes the positional grid (plus class token)."""
    state = {k: v for k, v in state.items() if not k.startswith('head.')}
    state['pos_embed'] = resize_positional_embedding(state['pos_embed'][0], grid)[None]
    return state


def timm_dinov2_state(state, grid):
    """timm DINOv2 checkpoint -> DINOv2VisionTransformer keys, positional grid resized."""
    state = {k: v for k, v in state.items() if k != 'mask_token'}
    state['pos_embed'] = resize_positional_embedding(state['pos_embed'][0], grid)[None]
    return state

def resize_positional_embedding(embedding, grid):
    """Bicubic resize of a square patch grid (plus class token) to `grid` (rows, cols)."""
    cls, patches = embedding[:1], embedding[1:]
    side = int(math.isqrt(patches.shape[0]))
    if side * side != patches.shape[0]:
        raise ValueError('Pretrained positional embedding is not a square grid')
    if (side, side) == tuple(grid):
        return embedding
    patches = patches.reshape(1, side, side, -1).permute(0, 3, 1, 2)
    # Same as timm's resample_abs_pos_embed (bicubic, antialiased when shrinking).
    patches = F.interpolate(patches.float(), size=tuple(grid), mode='bicubic', antialias=True)
    return torch.cat([cls, patches.permute(0, 2, 3, 1).reshape(grid[0] * grid[1], -1).to(embedding.dtype)])


def openai_visual_state(state, grid):
    """OpenAI CLIP checkpoint -> CLIPVisionTransformer keys; drops text tower and projection."""
    visual = {k[len('visual.'):]: v for k, v in state.items() if k.startswith('visual.') and k != 'visual.proj'}
    visual['positional_embedding'] = resize_positional_embedding(visual['positional_embedding'], grid)
    return visual


def build_vit(architecture, image_size, num_classes=40, dropout=0.0, query_head=False):
    family = VIT_SPECS[architecture].get('family')
    model_class = {'dinov2': DINOv2VisionTransformer, 'pe': PEVisionTransformer}.get(family, CLIPVisionTransformer)
    return model_class(**VIT_SPECS[architecture], image_size=tuple(image_size), num_classes=num_classes,
                       dropout=dropout, query_head=query_head)
