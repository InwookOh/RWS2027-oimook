"""15% attribute-only selection, then 90% MASK / 10% unchanged."""
import torch

IGNORE_INDEX = -100


def mask_attribute_tokens(token_ids, attribute_token_mask, mask_id, generator=None):
    tokens = torch.as_tensor(token_ids, dtype=torch.long).clone()
    eligible = torch.as_tensor(attribute_token_mask, dtype=torch.bool)
    if tokens.shape != eligible.shape:
        raise ValueError('Attribute token mask and token IDs must have equal shapes')
    eligible = eligible & tokens.ne(0) & tokens.ne(49406) & tokens.ne(49407) & tokens.ne(mask_id)
    selected = eligible & (torch.rand(tokens.shape, generator=generator) < 0.15)
    replacement = selected & (torch.rand(tokens.shape, generator=generator) < 0.90)
    labels = torch.full_like(tokens, IGNORE_INDEX)
    labels[selected] = tokens[selected]
    tokens[replacement] = mask_id
    return tokens, labels
