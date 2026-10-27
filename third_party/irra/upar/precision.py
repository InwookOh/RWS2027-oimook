import torch


def configure_fp32():
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def assert_model_fp32(model, gradients=False):
    try:
        cpu_amp = torch.is_autocast_enabled('cpu')
    except TypeError:
        cpu_amp = torch.is_autocast_cpu_enabled()
    if torch.is_autocast_enabled() or cpu_amp:
        raise RuntimeError('AMP is forbidden')
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_floating_point() and value.dtype != torch.float32:
            raise TypeError('Non-FP32 tensor: ' + name)
    if gradients:
        for name, p in model.named_parameters():
            if p.grad is not None and (p.grad.dtype != torch.float32 or not torch.isfinite(p.grad).all()):
                raise FloatingPointError('Invalid gradient: ' + name)


def assert_optimizer_fp32(optimizer):
    for values in optimizer.state.values():
        for value in values.values():
            if torch.is_tensor(value) and value.is_floating_point() and value.dtype != torch.float32:
                raise TypeError('Optimizer state must be FP32')
