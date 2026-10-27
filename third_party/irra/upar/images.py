"""Image IO and shared CLIP transforms, without training-only dependencies."""
from PIL import Image
from datasets.transforms import build_transforms


def image_transform(training=False):
    # DESIGN.md permits resize, horizontal flip during training, and normalization.
    return build_transforms(img_size=(378, 126), aug=False, is_train=training)


def read_rgb(path):
    with Image.open(path) as image:
        return image.convert('RGB')
