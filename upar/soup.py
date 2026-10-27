"""Weight-space mixing of checkpoints fine-tuned from the same initialization.

A uniform soup averages several fine-tuned models (Wortsman et al., ICML 2022); WiSE-FT
interpolates between the pretrained-backbone model and a fine-tuned one (Wortsman et al.,
CVPR 2022). Both give one model of the original size.
"""
import torch

from .config import model_options
from .model import build_model


def load_checkpoint(path):
    # Training artifacts contain only tensors and primitive Python values.
    return torch.load(path, map_location='cpu', weights_only=True)


def mix_checkpoints(checkpoints, weights, sources=None):
    """Checkpoint whose float tensors are sum(weight * member tensor); other tensors must match."""
    if not checkpoints or len(checkpoints) != len(weights):
        raise ValueError('Need one weight per checkpoint')
    if abs(sum(weights) - 1) > 1e-6:
        raise ValueError(f'Mixing weights must sum to 1, got {sum(weights)}')
    first = checkpoints[0]
    for checkpoint in checkpoints[1:]:
        if (checkpoint['architecture'] != first['architecture'] or checkpoint['preprocess'] != first['preprocess']
                or checkpoint['model_state'].keys() != first['model_state'].keys()):
            raise ValueError('Mixed checkpoints must share architecture, input size and parameter names')
    state = {}
    for key, value in first['model_state'].items():
        if value.is_floating_point():
            mixed = sum(w * c['model_state'][key].double() for c, w in zip(checkpoints, weights))
            state[key] = mixed.to(torch.float32)
        elif all(torch.equal(c['model_state'][key], value) for c in checkpoints):
            state[key] = value
        else:
            raise ValueError(f'Non-float tensor {key} differs between checkpoints')
    return {**first, 'model_state': state, 'epoch': None, 'validation_mADM': None,
            'mixture': {'sources': [str(s) for s in sources] if sources else None, 'weights': list(weights)}}


def wise_weights(alpha, members, base=True):
    """(1 - alpha) on the base checkpoint, alpha split evenly over the fine-tuned members."""
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0, 1]')
    soup = [alpha / members] * members
    return [1 - alpha] + soup if base else soup


def model_from_checkpoint(checkpoint, device):
    model = build_model(checkpoint['architecture'], pretrained=False, **model_options(checkpoint.get('training_config', {})))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in checkpoint['model_state'].items()}, strict=True)
    return model.to(device).eval()
