"""Training helpers: label smoothing, the DeepMAR loss and the weight EMA."""
from copy import deepcopy

import torch


def smooth_targets(labels, amount):
    if amount == 0:
        return labels
    return labels * (1 - amount) + 0.5 * amount


class DeepMARLoss(torch.nn.Module):
    """Weighted sigmoid cross-entropy of DeepMAR (Li et al., 2015), as in the Jia et al. strong baseline.

    Positives of attribute l are weighted exp(1 - r_l) and negatives exp(r_l), where r_l is the
    training positive rate. Weights use hard labels, so label smoothing may be applied to targets.
    """

    def __init__(self, positive_rate):
        super().__init__()
        self.register_buffer('positive_rate', torch.as_tensor(positive_rate, dtype=torch.float32))

    def forward(self, logits, targets):
        hard = (targets > 0.5).to(logits.dtype)
        rate = self.positive_rate.to(logits.dtype)
        weights = torch.exp(hard * (1 - rate) + (1 - hard) * rate)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, weight=weights)


class ModelEMA:
    """Exponential moving average of the weights (evaluated and saved instead of the live model)."""

    def __init__(self, model, decay):
        self.model = deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        for key, target in self.model.state_dict().items():
            if target.is_floating_point():
                target.mul_(self.decay).add_(source[key], alpha=1 - self.decay)
            else:
                target.copy_(source[key])
        self.num_updates += 1
