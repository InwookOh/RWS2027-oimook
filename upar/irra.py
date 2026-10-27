"""A teammate's IRRA-QB FrozenMultiScaleLocalAttr model as an ensemble member.

OpenAI CLIP ViT-L/14 at 378x126 fine-tuned with IRRA losses (SDM image-text matching and
masked attribute prediction), then frozen; patch tokens after blocks 8 and 16 (LayerNorm and a
1024 -> 768 projection) and the final projected tokens are mixed with softmax weights, and 40
attribute queries cross-attend to them (8 heads) to give one logit per attribute. Only this
attribute branch gives the member's sigmoid probabilities. With the text tower (converted with
--text), the original CLIP image-text score is also available: the cosine between the normalized
global image embedding (class token after ln_post and the projection) and the text embedding of
the query's template sentence (`irra_prompt`, ported from the original upar/prompts.py).
Plain PyTorch names follow the original (converted by scripts/irra_member.py).
"""
import io

import torch
from torch import nn
import torch.nn.functional as F

from .vit import VIT_SPECS, CLIPVisionTransformer

COLORS = ('Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow', 'Other')


def _join(parts, separator=' and '):
    return separator.join(parts)


def _clothing(v, upper):
    """Clothing phrase of the original prompt template (upar/prompts.py, upar-sentence-v1)."""
    length_index, offset = (7, 8) if upper else (20, 21)
    values = ['short'] if v[length_index] else []
    colors = _join([c.lower() if c != 'Other' else 'other-colored' for i, c in enumerate(COLORS) if v[offset + i]])
    if colors:
        values.append(colors)
    words = ' '.join(values)
    types = []
    if not upper:
        if v[33]:
            types.append('trousers or shorts')
        if v[34]:
            types.append('skirts or dresses')
    if not words and not types:
        return ''
    if types:
        return (words + ' ' if words else '') + _join(types)
    if upper:
        first = next((i for i in range(8, 20) if v[i]), None)
        article = 'an ' if not v[7] and first in (13, 19) else 'a '
        return article + words + ' upper garment'
    return words + ' lower clothing'


def irra_prompt(v):
    """The teammate's deterministic positive-only sentence for a binary attribute vector."""
    v = [int(x) for x in v]
    clauses = []
    ages = [w for i, w in enumerate(('young', 'adult', 'old')) if v[i]]
    if ages or v[3]:
        if len(ages) == 1:
            article = 'an ' if ages[0] == 'adult' else 'a '
            clauses.append('is ' + article + ages[0] + ' ' + ('woman' if v[3] else 'person'))
        elif not ages:
            clauses.append('is a woman')
        else:
            clauses.append('is ' + _join(ages) + (' and female' if v[3] else ''))
    hair = [w for i, w in ((4, 'short hair'), (5, 'long hair')) if v[i]]
    if hair:
        clauses.append('has ' + _join(hair))
    if v[6]:
        clauses.append('is bald')
    clothing = [c for c in (_clothing(v, True), _clothing(v, False)) if c]
    if clothing:
        clauses.append('wears ' + _join(clothing))
    carrying = ['a ' + w for i, w in ((35, 'backpack'), (36, 'bag')) if v[i]]
    if carrying:
        clauses.append('carries ' + _join(carrying))
    wearing = [('a ' if i == 39 else '') + w for i, w in ((37, 'regular glasses'), (38, 'sunglasses'), (39, 'hat'))
               if v[i]]
    if wearing:
        clauses.append('wears ' + _join(wearing))
    if not clauses:
        return 'The pedestrian is visible.'
    text = 'The pedestrian '
    for index, clause in enumerate(clauses):
        if index:
            text += ' and ' if len(clauses) == 2 else ', and ' if index == len(clauses) - 1 else ', '
        text += clause
    return text + '.'


BLOCKS = (8, 16)  # 1-based transformer blocks whose patch tokens feed the fusion


class WeightedPatchFusion(nn.Module):
    def __init__(self, width=1024, dim=768):
        super().__init__()
        self.norms = nn.ModuleDict({str(b): nn.LayerNorm(width) for b in BLOCKS})
        self.projections = nn.ModuleDict({str(b): nn.Linear(width, dim) for b in BLOCKS})
        self.fusion_logits = nn.Parameter(torch.zeros(3))

    def forward(self, intermediate, final_patches):
        features = [self.projections[str(b)](self.norms[str(b)](intermediate[b])) for b in BLOCKS]
        features.append(final_patches)
        weights = self.fusion_logits.softmax(0)
        return sum(weights[i].to(f.dtype) * f for i, f in enumerate(features))


class LocalAttributeBranch(nn.Module):
    def __init__(self, dim=768, heads=8, attributes=40):
        super().__init__()
        self.attribute_queries = nn.Parameter(torch.zeros(attributes, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.attribute_weights = nn.Parameter(torch.zeros(attributes, dim))
        self.attribute_bias = nn.Parameter(torch.zeros(attributes))

    def forward(self, patches):
        queries = self.query_norm(self.attribute_queries).unsqueeze(0).expand(len(patches), -1, -1)
        patches = self.patch_norm(patches)
        features, _ = self.cross_attention(queries.to(patches.dtype), patches, patches, need_weights=False)
        features = self.output_norm(features)
        return (torch.einsum('bkd,kd->bk', features.float(), self.attribute_weights.float())
                + self.attribute_bias.float())


class IRRAMember(nn.Module):
    def __init__(self, image_size=(378, 126), text=False):
        super().__init__()
        # Optional fine-tuned text tower for the CLIP image-text score (inference.py, `irra_text`).
        if text:
            from .clip_text import TEXT_SPECS, CLIPTextTransformer
            self.text = CLIPTextTransformer(**TEXT_SPECS['clip_openai_l14'])
        self.keep_global = False  # store the normalized global image embedding of the last batch
        self.visual = CLIPVisionTransformer(**VIT_SPECS['clip_vit_l14'], image_size=tuple(image_size))
        del self.visual.head
        self.proj = nn.Parameter(torch.zeros(1024, 768))  # CLIP's image projection
        self.multiscale = WeightedPatchFusion()
        self.local_branch = LocalAttributeBranch()
        self.head_prefix = 'local_branch.'

    def forward(self, x):
        visual = self.visual
        x = visual.conv1(x).flatten(2).transpose(1, 2)
        x = torch.cat([visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], dim=1)
        x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
        intermediate = {}
        for index, block in enumerate(visual.transformer.resblocks, start=1):
            x = block(x)
            if index in BLOCKS:
                intermediate[index] = x[:, 1:]
        final = visual.ln_post(x) @ self.proj.to(x.dtype)
        if self.keep_global:
            self.last_global = F.normalize(final[:, 0].float(), dim=-1)
        return self.local_branch(self.multiscale(intermediate, final[:, 1:]))

    @torch.no_grad()
    def query_features(self, queries, bpe, batch_size=64):
        """Normalized text embeddings of the queries' prompts (bpe: gzipped CLIP vocabulary bytes)."""
        from .clip_text import Tokenizer
        tokenizer = Tokenizer(io.BytesIO(bpe))
        tokens = tokenizer([irra_prompt(q) for q in queries]).to(next(self.text.parameters()).device)
        parts = [self.text(tokens[i:i + batch_size]).float() for i in range(0, len(tokens), batch_size)]
        return F.normalize(torch.cat(parts), dim=-1)
