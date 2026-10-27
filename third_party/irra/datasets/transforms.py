"""IRRA image preprocessing shared by original retrieval and UPAR."""
import torchvision.transforms as T


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    if not is_train:
        return T.Compose([T.Resize((height, width)), T.ToTensor(),
                          T.Normalize(mean=mean, std=std)])
    if aug:
        return T.Compose([
            T.Resize((height, width)), T.RandomHorizontalFlip(0.5), T.Pad(10),
            T.RandomCrop((height, width)), T.ToTensor(), T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    return T.Compose([T.Resize((height, width)), T.RandomHorizontalFlip(0.5),
                      T.ToTensor(), T.Normalize(mean=mean, std=std)])
