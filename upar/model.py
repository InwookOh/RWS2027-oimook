from .vit import VIT_SPECS, build_vit


ARCHITECTURES = (*VIT_SPECS, 'e1_clip_l14', 'irra_clip_l14')


def build_model(architecture, pretrained=False, dropout=0.0, image_size=(256, 128), query_head=False, irra_text=False):
    if architecture == 'e1_clip_l14':  # a teammate's E1 model, converted by scripts/e1_member.py
        from .e1 import E1Member
        return E1Member(image_size)
    if architecture == 'irra_clip_l14':  # a teammate's IRRA-QB model, converted by scripts/irra_member.py
        from .irra import IRRAMember
        return IRRAMember(image_size, text=irra_text)
    if architecture not in VIT_SPECS:
        raise ValueError(f'Unsupported architecture: {architecture}')
    if pretrained:
        raise ValueError('Backbone weights come from upar/pretrained.py, not torchvision')
    return build_vit(architecture, image_size, dropout=dropout, query_head=query_head)
