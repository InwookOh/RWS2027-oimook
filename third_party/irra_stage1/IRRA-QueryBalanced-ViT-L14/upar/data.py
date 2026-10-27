"""UPAR CSV splits, semantic query joins, and image/query datasets."""
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .sampling import QueryBalancedSampler

from .attributes import ATTRIBUTE_NAMES, attribute_vector
from .images import image_transform, read_rgb
from .masking import mask_attribute_tokens
from .prompts import encode_query


def _rows(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
        for number, row in enumerate(csv.reader(stream), start=1):
            if row and any(cell.strip() for cell in row):
                yield number, [cell.strip() for cell in row]


def read_attribute_csv(path, with_images=False):
    """Read official comment header and binary vectors without relabelling rows."""
    rows = iter(_rows(path))
    try:
        _, header = next(rows)
    except StopIteration:
        raise ValueError(f'Empty attribute CSV: {path}') from None
    header[0] = header[0].lstrip('#').strip()
    expected = ('image', *ATTRIBUTE_NAMES) if with_images else ATTRIBUTE_NAMES
    if tuple(header) != expected:
        raise ValueError(f'Invalid attribute header/order in {path}')
    entries = []
    for number, row in rows:
        offset = int(with_images)
        if len(row) != 40 + offset:
            raise ValueError(f'{path}:{number}: expected {40 + offset} columns')
        if any(value not in ('0', '1') for value in row[offset:]):
            raise ValueError(f'{path}:{number}: attributes must be 0 or 1')
        vector = attribute_vector(tuple(int(value) for value in row[offset:]))
        entries.append((row[0], vector) if with_images else vector)
    return tuple(entries)


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    image_path: Path
    attributes: tuple
    query_index: int


@dataclass(frozen=True)
class UPARSplit:
    name: str
    images: tuple
    queries: tuple

    @classmethod
    def read(cls, data_root, split):
        if split not in ('train', 'val'):
            raise ValueError('UPAR public splits are train and val')
        root = Path(data_root).resolve()
        directory = root / 'annotations' / 'task2' / split
        queries = read_attribute_csv(directory / 'queries.csv')
        if not queries or len(set(queries)) != len(queries):
            raise ValueError(f'{split}: queries must be distinct, nonempty attribute vectors')
        ground_truth = read_attribute_csv(directory / 'gt.csv', with_images=True)
        ids = {}
        for number, row in _rows(directory / 'ids.csv'):
            if row[0].startswith('#'):
                continue
            if len(row) != 2:
                raise ValueError(f'ids.csv:{number}: expected image path and semantic query index')
            path, index = row[0], int(row[1])
            if path in ids or not 0 <= index < len(queries):
                raise ValueError(f'{split}: invalid semantic query mapping for {path}')
            ids[path] = index
        images, seen, referenced = [], set(), set()
        for relative_path, vector in ground_truth:
            if relative_path in seen or relative_path not in ids:
                raise ValueError(f'{split}: duplicate image or missing semantic mapping: {relative_path}')
            seen.add(relative_path)
            query_index = ids[relative_path]
            if queries[query_index] != vector:
                raise ValueError(f'{split}: gt vector differs from queries[ids[image]]: {relative_path}')
            relative = Path(relative_path)
            image_path = (root / relative).resolve()
            if relative.is_absolute():
                raise ValueError(f'Image path must be relative to data/: {relative_path}')
            try:
                image_path.relative_to(root)
            except ValueError:
                # Path.is_relative_to() was added in Python 3.9. Keep this
                # containment check compatible with the Python 3.7 training image.
                raise ValueError(
                    f'Image path must be relative to data/: {relative_path}') from None
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            images.append(ImageRecord(relative_path, image_path, vector, query_index))
            referenced.add(query_index)
        if seen != set(ids) or referenced != set(range(len(queries))):
            raise ValueError(f'{split}: gt/ids/queries do not describe the same gallery')
        return cls(split, tuple(images), queries)


class TrainingDataset(Dataset):
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.transform = image_transform(training=True)
        self.prompts = tuple(encode_query(query, tokenizer) for query in split.queries)

    def __len__(self):
        return len(self.split.images)

    def __getitem__(self, index):
        record = self.split.images[index]
        prompt = self.prompts[record.query_index]
        masked, labels = mask_attribute_tokens(
            prompt.token_ids, prompt.attribute_token_mask, self.tokenizer.mask_id)
        return {
            'images': self.transform(read_rgb(record.image_path)),
            'attributes': torch.tensor(record.attributes, dtype=torch.long),
            'caption_ids': torch.tensor(prompt.token_ids, dtype=torch.long),
            'map_ids': masked,
            'map_labels': labels,
        }


class GalleryDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.transform = image_transform()

    def __len__(self):
        return len(self.split.images)

    def __getitem__(self, index):
        record = self.split.images[index]
        return (torch.tensor(record.attributes, dtype=torch.long),
                self.transform(read_rgb(record.image_path)))


class QueryDataset(Dataset):
    def __init__(self, split, tokenizer):
        self.queries = split.queries
        self.prompts = tuple(encode_query(query, tokenizer) for query in split.queries)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        return (torch.tensor(self.queries[index], dtype=torch.long),
                torch.tensor(self.prompts[index].token_ids, dtype=torch.long))


def validation_loaders(split, tokenizer, batch_size=512, num_workers=8):
    return (DataLoader(GalleryDataset(split), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers),
            DataLoader(QueryDataset(split, tokenizer), batch_size=batch_size, shuffle=False,
                       num_workers=num_workers))


def training_loaders(data_root, tokenizer, config):
    train = UPARSplit.read(data_root, 'train')
    validation = UPARSplit.read(data_root, 'val')
    loader = DataLoader(TrainingDataset(train, tokenizer), batch_size=config.batch_size,
                        sampler=QueryBalancedSampler(train, config.seed),
                        num_workers=config.num_workers,
                        generator=torch.Generator().manual_seed(config.seed))
    gallery, queries = validation_loaders(validation, tokenizer, config.test_batch_size,
                                         config.num_workers)
    return loader, gallery, queries
