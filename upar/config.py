"""Training configuration: defaults and validation."""
from copy import deepcopy
import math

from .data import DOMAINS

# Checkpoints whose CLIP text tower upar/clip_text.py can build (TEXT_SPECS; a test keeps the two
# in sync). Listed here because config.py ships in submissions and clip_text.py does not.
TEXT_PRIOR_SOURCES = ('clip_openai_l14', 'clip_openai_l14_336', 'clip_metaclip_l14')
from .model import ARCHITECTURES

LOSSES = ('bce', 'weighted_bce', 'deepmar_bce')


def normalize_config(config):
    config = deepcopy(config)
    backbone = config.get('backbone', config.get('architecture'))
    if 'backbone' in config and 'architecture' in config and config['architecture'] != backbone:
        raise ValueError('backbone and legacy architecture must agree')
    if backbone not in ARCHITECTURES:
        raise ValueError(f'Unsupported backbone: {backbone}')
    config['backbone'] = config['architecture'] = backbone
    defaults = {'amp': False, 'prediction_cache': True, 'ema': False, 'ema_decay': 0.999, 'label_smoothing': 0.0,
                'retrieval': 'l1', 'dropout': 0.0, 'augmix': False,
                'save_epoch_checkpoints': False, 'linear_probe_epochs': 0, 'linear_probe_lr': 1e-3,
                'train_with_val': False, 'head_init': None, 'head_init_l2': 0.1,
                'head_init_images_per_domain': 3000, 'query_head': False}
    for key, value in defaults.items():
        config.setdefault(key, value)
    for key in ('ema', 'amp', 'prediction_cache', 'augmix', 'save_epoch_checkpoints', 'train_with_val', 'query_head'):
        if not isinstance(config[key], bool):
            raise ValueError(f'{key} must be a JSON boolean')
    for key in ('ema_decay', 'label_smoothing', 'dropout'):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number')
    if not 0 <= config['ema_decay'] < 1:
        raise ValueError('ema_decay must be in [0, 1)')
    if not 0 <= config['label_smoothing'] < 1:
        raise ValueError('label_smoothing must be in [0, 1)')
    if not 0 <= config['dropout'] < 1:
        raise ValueError('dropout must be in [0, 1)')
    if config.get('loss', 'bce') not in LOSSES:
        raise ValueError('Unsupported loss')
    from .retrieval import RETRIEVAL_METHODS
    if config['retrieval'] not in RETRIEVAL_METHODS:
        raise ValueError(f'retrieval must be one of {RETRIEVAL_METHODS}')
    if type(config['linear_probe_epochs']) is not int or config['linear_probe_epochs'] < 0:
        raise ValueError('linear_probe_epochs must be a nonnegative integer')
    lr = config['linear_probe_lr']
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('linear_probe_lr must be finite and positive')
    if config['head_init'] not in (None, 'text_prior'):
        raise ValueError('head_init must be null or "text_prior"')
    if config['head_init'] == 'text_prior' and config.get('pretrained') not in TEXT_PRIOR_SOURCES:
        raise ValueError('head_init "text_prior" needs a CLIP ViT-L/14 checkpoint with a text tower')
    if config['head_init'] == 'text_prior' and config['linear_probe_epochs'] and not config['query_head']:
        # With a text-prior head the probing epochs train only the query head (train.py).
        raise ValueError('linear_probe_epochs with head_init "text_prior" needs query_head (nothing else to probe)')
    if type(config['head_init_images_per_domain']) is not int or config['head_init_images_per_domain'] < 1:
        raise ValueError('head_init_images_per_domain must be a positive integer')
    config.setdefault('holdout_domain', None)
    if config['holdout_domain'] is not None and config['holdout_domain'] not in DOMAINS:
        raise ValueError(f'holdout_domain must be null or one of {DOMAINS}')
    return config


def model_options(config):
    """build_model arguments from a (training) config; also reads the configs stored in older
    checkpoints, whose extra keys are ignored."""
    preprocess = config.get('preprocess', {})
    return {'dropout': config.get('dropout', 0.0), 'query_head': config.get('query_head', False),
            'image_size': (int(preprocess.get('height', 256)), int(preprocess.get('width', 128))),
            'irra_text': bool(config.get('irra_text', False))}
