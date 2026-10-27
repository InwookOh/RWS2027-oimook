"""Only plain model.state_dict(), only completed 1-based epochs 1 through 10."""
import os
from pathlib import Path
import uuid
import torch


def save_epoch_weights(model, output, epoch_number):
    if type(epoch_number) is not int or not 1 <= epoch_number <= 60:
        raise ValueError('Expected a 1-based Epoch in 1..60')
    if epoch_number > 10: return None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / ('epoch_%03d.pth' % epoch_number)
    if target.exists(): raise FileExistsError('Never overwrite weights: ' + str(target))
    state = model.state_dict()
    if any(not torch.is_tensor(v) or (v.is_floating_point() and v.dtype != torch.float32) for v in state.values()):
        raise TypeError('Only FP32 model state_dict tensors may be saved')
    temporary = output / ('.weights-tmp-' + uuid.uuid4().hex)
    try:
        torch.save(state, str(temporary))
        with temporary.open('rb') as stream: os.fsync(stream.fileno())
        os.link(str(temporary), str(target))  # exclusive and atomic publication
    finally:
        if temporary.exists(): temporary.unlink()
    return target


def load_state_dict(path):
    try:
        state = torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError as error:
        if 'weights_only' not in str(error): raise
        state = torch.load(str(path), map_location='cpu')
    if not isinstance(state, dict) or not state or any(not torch.is_tensor(v) for v in state.values()):
        raise ValueError('Only plain model weights are accepted')
    return state
