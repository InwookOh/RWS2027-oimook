import csv
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .attributes import ATTRIBUTES


def read_binary_csv(path, with_paths=False):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = (['image'] if with_paths else []) + list(ATTRIBUTES)
        actual = [s.strip() for s in header]
        actual[0] = actual[0].lstrip('#').strip()
        if actual != expected:
            raise ValueError(f'{path}: unexpected attribute header/order')
        paths, values = [], []
        for row in reader:
            if not row:
                continue
            if len(row) != len(expected):
                raise ValueError(f'{path}: invalid column count')
            attrs = row[1:] if with_paths else row
            if any(v not in ('0', '1') for v in attrs):
                raise ValueError(f'{path}: attributes must be binary')
            values.append([int(v) for v in attrs])
            if with_paths:
                paths.append(row[0])
    matrix = np.asarray(values, dtype=np.float32).reshape(-1, 40)
    if not len(matrix):
        raise ValueError(f'{path}: empty annotation')
    return paths, matrix


def load_split(data_dir, split, limit=None):
    folder = Path(data_dir) / 'annotations' / 'task2' / split
    paths, labels = read_binary_csv(folder / 'gt.csv', with_paths=True)
    _, queries = read_binary_csv(folder / 'queries.csv')
    with (folder / 'ids.csv').open(newline='') as f:
        pairs = [r for r in csv.reader(f) if r]
    mapping = dict(pairs)
    if len(set(paths)) != len(paths) or len(mapping) != len(pairs):
        raise ValueError('Duplicate image paths in annotations')
    if set(mapping) != set(paths):
        raise ValueError('gt.csv and ids.csv must contain the same images')
    ids = np.asarray([int(mapping[p]) for p in paths], dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= len(queries)):
        raise ValueError('Query ID outside queries.csv')
    if not np.array_equal(labels, queries[ids]):
        raise ValueError('Image labels do not match their query IDs')
    if limit is not None:
        if limit < 1:
            raise ValueError('Image limit must be positive')
        paths, labels, ids = paths[:limit], labels[:limit], ids[:limit]
    return paths, labels, queries, ids


DOMAINS = ('Market1501', 'PA100k', 'PETA')


def image_domains(paths):
    return np.asarray([p.split('/')[0] for p in paths])


def select_domains(paths, labels, ids, keep):
    """Rows whose source dataset is in `keep`, preserving annotation order."""
    mask = np.isin(image_domains(paths), list(keep))
    if not mask.any():
        raise ValueError(f'No images from domains {sorted(keep)}')
    return [p for p, m in zip(paths, mask) if m], labels[mask], ids[mask]


def make_transform(preprocess, training=False, augmix=False, normalize=True):
    height, width = int(preprocess['height']), int(preprocess['width'])
    if height < 32 or width < 32:
        raise ValueError('Image dimensions must be at least 32')
    # Bilinear for our models; a converted member may ask for its own resize (E1: bicubic).
    interpolation = {'bilinear': InterpolationMode.BILINEAR, 'bicubic': InterpolationMode.BICUBIC}[
        preprocess.get('interpolation', 'bilinear')]
    steps = [T.Resize((height, width), interpolation=interpolation)]
    if training:
        steps += [T.Pad(10), T.RandomCrop((height, width)), T.RandomHorizontalFlip()]
        if augmix:
            steps.append(T.AugMix())
    steps.append(T.ToTensor())
    if normalize:
        steps.append(T.Normalize(preprocess['mean'], preprocess['std']))
    return T.Compose(steps)


class ImageDataset(Dataset):
    def __init__(self, paths, transform, labels=None):
        self.paths, self.transform, self.labels = paths, transform, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as im:
            image = self.transform(im.convert('RGB'))
        if self.labels is None:
            return image
        return image, self.labels[index]
