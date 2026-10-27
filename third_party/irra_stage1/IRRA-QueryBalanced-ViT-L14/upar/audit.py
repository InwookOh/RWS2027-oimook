import hashlib
from pathlib import Path
from collections import Counter
from .sampling import QueryBalancedSampler


def audit_splits(train, val, root=None):
    a, b = {r.relative_path for r in train.images}, {r.relative_path for r in val.images}
    ap, bp = {str(r.image_path.resolve()) for r in train.images}, {str(r.image_path.resolve()) for r in val.images}
    if a & b or ap & bp: raise ValueError('Train/Validation sample leakage')
    if len(a) != len(train.images) or len(b) != len(val.images): raise ValueError('Duplicate Sample IDs')
    sampler = QueryBalancedSampler(train, 1)
    digest = lambda split: hashlib.sha256(b''.join(r.relative_path.encode() + b'\0' + bytes(r.attributes)
                                                  for r in split.images)).hexdigest()
    value = dict(train_images=len(train.images), train_queries=len(train.queries),
                 validation_images=len(val.images), validation_queries=len(val.queries),
                 train_domains=dict(Counter(Path(r.relative_path).parts[0] for r in train.images)),
                 train_validation_id_overlap=0, train_validation_resolved_path_overlap=0,
                 train_fingerprint=digest(train), validation_fingerprint=digest(val),
                 sampling=sampler.metadata(), all_image_files_checked=True,
                 pixel_content_deduplication_performed=False, test_labels_used=False)
    if root is not None:
        value['annotation_sha256'] = {split: {name: hashlib.sha256((Path(root)/'annotations/task2'/split/name).read_bytes()).hexdigest()
                                             for name in ('gt.csv', 'ids.csv', 'queries.csv')} for split in ('train','val')}
    return value
