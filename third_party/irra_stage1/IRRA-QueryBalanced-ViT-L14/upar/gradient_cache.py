"""Exact full-effective-batch SDM, token-mean MAP, and one optimizer update."""
from contextlib import contextmanager
import torch
from model.objectives import compute_attribute_sdm
from .precision import assert_model_fp32, assert_optimizer_fp32


def validate_cache_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise ValueError('BatchNorm changes batch decomposition')
        if isinstance(module, torch.nn.Dropout) and module.p:
            raise ValueError('Nonzero dropout is not supported in the exact batch decomposition')
        if isinstance(module, torch.nn.MultiheadAttention) and module.dropout:
            raise ValueError('Nonzero attention dropout is unsupported')
    if model.logit_scale.requires_grad: raise ValueError('Require baseline fixed temperature')


def rng_state(device):
    return torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None


@contextmanager
def replay_rng(state, device):
    with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
        torch.set_rng_state(state[0])
        if state[1] is not None: torch.cuda.set_rng_state(state[1], device)
        yield


def backward_effective_batch(model, batch, micro_batch_size):
    assert_model_fp32(model)
    validate_cache_model(model)
    if not model.training or micro_batch_size < 1: raise ValueError('Invalid cache training mode')
    device = next(model.parameters()).device
    n = len(batch['images'])
    if n < 1 or any(len(v) != n for v in batch.values()): raise ValueError('Invalid effective batch')
    chunks = [(i, min(i + micro_batch_size, n)) for i in range(0, n, micro_batch_size)]
    sub_batch = lambda a, b: {k: v[a:b].to(device) for k, v in batch.items()}
    cache_image, cache_text, random_states = [], [], []
    with torch.no_grad():
        for start, end in chunks:
            random_states.append(rng_state(device))
            sub = sub_batch(start, end)
            image, text, tokens = model.global_features(sub)
            if image.dtype != torch.float32 or text.dtype != torch.float32: raise TypeError('Features must be FP32')
            cache_image.append(image.detach())
            cache_text.append(text.detach())
            del sub, image, text, tokens
    image = torch.cat(cache_image).detach().requires_grad_(True)
    text = torch.cat(cache_text).detach().requires_grad_(True)
    del cache_image, cache_text
    sdm = compute_attribute_sdm(image, text, batch['attributes'].to(device), model.logit_scale)
    if not torch.isfinite(sdm): raise FloatingPointError('Non-finite full-batch SDM')
    sdm.backward()
    image_grad, text_grad = image.grad.detach(), text.grad.detach()
    sdm_value = float(sdm.detach())
    del sdm, image, text
    selected_count = int(batch['map_labels'].ne(-100).sum())
    map_value, correct = 0.0, 0
    for index, (start, end) in enumerate(chunks):
        sub = sub_batch(start, end)
        with replay_rng(random_states[index], device):
            image, text, tokens = model.global_features(sub)
            result = model.map_result(sub, tokens)
            count = int(result['map_tokens'])
            weight = count / selected_count if selected_count else 0.0
            surrogate = (image * image_grad[start:end]).sum() + (text * text_grad[start:end]).sum()
            loss = surrogate + result['map_loss'] * weight
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite MAP/replay loss')
            loss.backward()
            map_value += float(result['map_loss'].detach()) * weight
            correct += int(result['map_correct'])
        del sub, image, text, tokens, result, surrogate, loss
    assert_model_fp32(model, gradients=True)
    return dict(loss=sdm_value + map_value, sdm_loss=sdm_value, map_loss=map_value,
                map_acc=correct / selected_count if selected_count else 0.0,
                map_tokens=selected_count, effective_batch_size=n, sdm_batch_size=n,
                micro_batch_size=micro_batch_size, micro_batches=len(chunks), gradient_cache=True)


def optimizer_update(model, optimizer, batch, micro_batch_size):
    optimizer.zero_grad(set_to_none=True)
    result = backward_effective_batch(model, batch, micro_batch_size)
    optimizer.step()
    assert_optimizer_fp32(optimizer)
    result['optimizer_steps'] = 1
    return result
