"""FP32 block activation recomputation without AMP machinery (torch 1.12+)."""
import torch


class _Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, x):
        if x.dtype != torch.float32:
            raise TypeError('Activation must be FP32')
        ctx.block = block
        ctx.save_for_backward(x)
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = torch.cuda.get_rng_state(x.device) if x.is_cuda else None
        ctx.device = x.device
        return block(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        replay = x.detach().requires_grad_(True)
        cuda_devices = [x.device.index] if x.is_cuda else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            with torch.enable_grad():
                output = ctx.block(replay)
            torch.autograd.backward(output, grad_output)
        return None, replay.grad


def recompute_block(block, x):
    return _Recompute.apply(block, x) if x.requires_grad else block(x)
