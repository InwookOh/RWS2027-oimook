import torch
import torch.nn.functional as F


def compute_attribute_sdm(image_features, text_features, attributes, logit_scale, epsilon=1e-8):
    """SDM with exact 40-bit attribute equality; no PID or class labels."""
    if attributes.ndim != 2 or attributes.shape != (image_features.shape[0], 40):
        raise ValueError('SDM requires one 40-D attribute vector per image-text pair')
    labels = attributes[:, None, :].eq(attributes[None, :, :]).all(dim=-1)
    return compute_sdm_with_labels(image_features, text_features, labels, logit_scale, epsilon)


def compute_sdm_with_labels(image_features, text_features, labels, logit_scale, epsilon=1e-8):
    """IRRA's bidirectional SDM calculation with explicit positive labels."""
    if image_features.dtype != torch.float32 or text_features.dtype != torch.float32:
        raise TypeError('SDM features must be FP32')
    return _sdm_float32(image_features, text_features, labels, logit_scale, epsilon)


def _sdm_float32(image_features, text_features, labels, logit_scale, epsilon):
    image_norm = F.normalize(image_features.float(), dim=1)
    text_norm = F.normalize(text_features.float(), dim=1)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels = labels.float()
    i2t_targets = labels / labels.sum(dim=1, keepdim=True)
    t2i_labels = labels.t()
    t2i_targets = t2i_labels / t2i_labels.sum(dim=1, keepdim=True)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(i2t_targets + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(t2i_targets + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_map(scores, original_tokens):
    """Cross entropy for selected attribute BPE positions, including unchanged ones."""
    if original_tokens.numel() == 0:
        return scores.float().sum() * 0.0
    return F.cross_entropy(scores.float(), original_tokens.long())

