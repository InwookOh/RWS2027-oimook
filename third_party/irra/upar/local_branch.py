"""Forty attribute queries attending exclusively to projected image patches."""
import torch
from torch import nn


class LocalAttributeBranch(nn.Module):
    def __init__(self, dim=512, heads=8, attributes=40):
        super().__init__()
        if attributes != 40 or dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('Expected 40 attributes and a dimension divisible by heads')
        self.dim = dim
        self.attribute_queries = nn.Parameter(torch.empty(40, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.patch_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(dim)
        self.attribute_weights = nn.Parameter(torch.empty(40, dim))
        self.attribute_bias = nn.Parameter(torch.zeros(40))
        nn.init.normal_(self.attribute_queries, std=0.02)
        nn.init.normal_(self.attribute_weights, std=dim ** -0.5)

    def forward(self, patches, return_attention=False):
        if patches.ndim != 3 or patches.shape[1:] != (243, self.dim):
            raise ValueError('Local branch requires exactly 243 projected patch tokens; exclude CLS')
        queries = self.query_norm(self.attribute_queries).unsqueeze(0).expand(len(patches), -1, -1)
        patches = self.patch_norm(patches)
        features, attention = self.cross_attention(queries, patches, patches, need_weights=return_attention)
        features = self.output_norm(features)
        logits = torch.einsum('bkd,kd->bk', features, self.attribute_weights) + self.attribute_bias
        return logits.float(), features, attention
