"""Shared image transforms (PROJECT_SPEC §7, DECISION_LOG D008/D011).

Random augmentation is applied to the training split only (CONSTITUTION C5).
All values come from `configs/config.yaml` so every experiment uses the same profile.
"""
from torchvision import transforms as T


def _normalize(cfg):
    norm = cfg["data"]["normalization"]
    return T.Normalize(mean=norm["mean"], std=norm["std"])


def build_train_transform(cfg):
    size = cfg["project"]["image_size"]
    aug = cfg["data"]["augmentation"]
    return T.Compose([
        T.RandomResizedCrop(size, scale=tuple(aug["random_resized_crop_scale"])),
        T.RandomHorizontalFlip(p=aug["horizontal_flip_p"]),
        T.RandomRotation(degrees=aug["rotation_degrees"]),
        T.ColorJitter(**aug["color_jitter"]),
        T.ToTensor(),
        _normalize(cfg),
    ])


def build_eval_transform(cfg):
    """Deterministic transform for validation, test, and E4's un-augmented training data.

    RealWaste images are 524×524 squares, so a plain resize keeps the whole object in view.
    """
    size = cfg["project"]["image_size"]
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        _normalize(cfg),
    ])
