"""CLIP text encoder and BPE tokenizer in plain PyTorch (training-time only).

Used to turn attribute descriptions into CLIP text embeddings, for example to
initialize the attribute head. Parameter names follow OpenAI CLIP's state dict, so the
text tower loads from the same checkpoint as the image tower (`clip_openai_l14`).
The tokenizer follows OpenAI's SimpleTokenizer for plain ASCII text (no ftfy/regex).
"""
from functools import lru_cache
import gzip
import hashlib
import os
from pathlib import Path
import re
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F

from .vit import activation_layer

BPE_URL = 'https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz'
BPE_SHA256 = '924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a'
TEXT_SPECS = {'clip_openai_l14': {'width': 768, 'layers': 12, 'heads': 12, 'context': 77, 'vocab': 49408}}
TEXT_SPECS['clip_openai_l14_336'] = TEXT_SPECS['clip_metaclip_l14'] = TEXT_SPECS['clip_openai_l14']


def bpe_path(root='outputs/pretrained/clip_bpe'):
    path = Path(root) / 'bpe_simple_vocab_16e6.txt.gz'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        urllib.request.urlretrieve(BPE_URL, temporary)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != BPE_SHA256:
        raise ValueError(f'{path}: SHA256 {digest} does not match the OpenAI CLIP vocabulary')
    return path


@lru_cache()
def bytes_to_unicode():
    codes = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    chars = codes[:]
    extra = 0
    for b in range(256):
        if b not in codes:
            codes.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(codes, map(chr, chars)))


class Tokenizer:
    START, END = '<|startoftext|>', '<|endoftext|>'

    def __init__(self, path=None):
        merges = gzip.open(path or bpe_path()).read().decode('utf-8').split('\n')
        merges = [tuple(m.split()) for m in merges[1:49152 - 256 - 2 + 1]]
        vocab = list(bytes_to_unicode().values())
        vocab += [v + '</w>' for v in vocab] + [''.join(m) for m in merges] + [self.START, self.END]
        self.encoder = {token: index for index, token in enumerate(vocab)}
        self.ranks = {merge: index for index, merge in enumerate(merges)}
        self.byte_encoder = bytes_to_unicode()
        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + '</w>']
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))
            if best not in self.ranks:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1]); i += 2
                else:
                    merged.append(word[i]); i += 1
            word = merged
        self.cache[token] = word
        return word

    def encode(self, text):
        if not text.isascii():
            raise ValueError('Only ASCII text is supported')
        text = ' '.join(text.split()).lower()
        ids = []
        # OpenAI pattern with \p{L} -> letters and \p{N} -> single digits (identical for ASCII).
        for token in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+", text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            ids.extend(self.encoder[piece] for piece in self.bpe(token))
        return ids

    def __call__(self, texts, context=77):
        result = torch.zeros(len(texts), context, dtype=torch.long)
        for row, text in enumerate(texts):
            ids = [self.encoder[self.START]] + self.encode(text) + [self.encoder[self.END]]
            if len(ids) > context:
                raise ValueError(f'Text too long for {context} tokens: {text}')
            result[row, :len(ids)] = torch.tensor(ids)
        return result


class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(batch, tokens, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(x.transpose(1, 2).reshape(batch, tokens, width))


class TextBlock(nn.Module):
    def __init__(self, width, heads, activation='quick_gelu'):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = CausalAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), activation_layer(activation), nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class CLIPTextTransformer(nn.Module):
    def __init__(self, width, layers, heads, context, vocab, activation='quick_gelu'):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab, width)
        self.positional_embedding = nn.Parameter(torch.empty(context, width))
        self.resblocks = nn.ModuleList(TextBlock(width, heads, activation) for _ in range(layers))
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, width))

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.positional_embedding[:tokens.shape[1]]
        for block in self.resblocks:
            x = block(x)
        x = self.ln_final(x)
        # The end-of-text token has the largest id; its output is the sentence embedding.
        return x[torch.arange(len(tokens)), tokens.argmax(-1)] @ self.text_projection


def openai_text_state(state):
    """OpenAI CLIP checkpoint -> CLIPTextTransformer keys (the text tower only)."""
    result = {}
    for key, value in state.items():
        if key.startswith('transformer.resblocks.'):
            key = key[len('transformer.'):].replace('mlp.c_fc', 'mlp.0').replace('mlp.c_proj', 'mlp.2')
        elif key not in ('token_embedding.weight', 'positional_embedding', 'ln_final.weight', 'ln_final.bias',
                         'text_projection'):
            continue
        result[key] = value
    return result


def build_text_encoder(state, source='clip_openai_l14'):
    model = CLIPTextTransformer(**TEXT_SPECS[source])
    model.load_state_dict(openai_text_state(state), strict=True)
    return model.eval()
