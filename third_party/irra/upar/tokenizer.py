"""IRRA CLIP BPE with offsets and stdlib-only handling of generated ASCII text.

The controlled English prompt alphabet uses the same pretokenization rules as
CLIP. BPE merges, byte encoding, vocabulary and the IRRA mask slot are reused
from SimpleTokenizer. This does not provide normalization for arbitrary user
text; generated prompts must already have canonical whitespace.
"""
import re

from utils.simple_tokenizer import SimpleTokenizer, default_bpe


class PromptTokenizer(SimpleTokenizer):
    def __init__(self, bpe_path=None):
        self._init_bpe(bpe_path or default_bpe())
        self.pat = re.compile(
            r"<\|startoftext\|>|<\|mask\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\sa-z0-9]+",
            re.IGNORECASE,
        )
        self.sot_id = self.encoder['<|startoftext|>']
        self.eot_id = self.encoder['<|endoftext|>']
        self.mask_id = self.encoder['<|mask|>']

    def encode_with_offsets(self, text):
        if not text.isascii() or text != ' '.join(text.split()):
            raise ValueError('Generated prompts must use canonical ASCII whitespace')
        ids, offsets = [], []
        # Lowercasing generated ASCII leaves all character offsets unchanged.
        for match in self.pat.finditer(text.lower()):
            token = match.group()
            if token in self.cache and token.startswith('<|'):
                ids.append(self.encoder[token])
                offsets.append((match.start(), match.end()))
                continue
            encoded = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            cursor = match.start()
            for piece in self.bpe(encoded).split(' '):
                surface = piece[:-4] if piece.endswith('</w>') else piece
                end = cursor + len(surface)
                ids.append(self.encoder[piece])
                offsets.append((cursor, end))
                cursor = end
            if cursor != match.end():
                raise ValueError('BPE offset alignment failed')
        return ids, offsets

    def encode(self, text):
        return self.encode_with_offsets(text)[0]
