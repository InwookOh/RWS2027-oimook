"""Train-only exact 40-bit groups; independent replacement draws per epoch."""
from collections import Counter
import hashlib
import torch
from torch.utils.data import Sampler


class QueryBalancedSampler(Sampler):
    replacement = True
    exponent = 0.5

    def __init__(self, split, seed=1):
        if split.name != 'train':
            raise ValueError('Sampling counts must come exclusively from train')
        self.vectors = tuple(tuple(record.attributes) for record in split.images)
        if not self.vectors or any(len(v) != 40 or any(x not in (0, 1) for x in v)
                                   for v in self.vectors):
            raise ValueError('Expected nonempty binary 40-D training vectors')
        self.counts = Counter(self.vectors)
        self.weights = torch.tensor([self.counts[v] ** -0.5 for v in self.vectors],
                                    dtype=torch.double)
        self.num_samples = len(self.vectors)
        self.seed = int(seed)
        self.epoch = 1
        self.fingerprint = hashlib.sha256(b''.join(
            record.relative_path.encode() + b'\0' + bytes(record.attributes)
            for record in split.images)).hexdigest()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch - 1)
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True,
                                      generator=generator).tolist())

    def __len__(self):
        return self.num_samples

    def metadata(self):
        return {'type': 'query_balanced', 'exponent': self.exponent,
                'replacement': True, 'seed': self.seed, 'num_samples': self.num_samples,
                'groups': len(self.counts), 'train_fingerprint': self.fingerprint,
                'epoch_seed_rule': 'seed + epoch - 1'}
