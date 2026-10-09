from torchvision import transforms as T


def build_train_transform(cfg):
    aug = cfg["data"]["augmentation"]
    norm = cfg["data"]["normalization"]
    return T.Compose([
        T.RandomResizedCrop(cfg["project"]["image_size"], scale=tuple(aug["random_resized_crop_scale"])),
        T.RandomHorizontalFlip(aug["horizontal_flip_p"]),
        T.RandomRotation(aug["rotation_degrees"]),
        T.ColorJitter(**aug["color_jitter"]),
        T.ToTensor(),
        T.Normalize(norm["mean"], norm["std"]),
    ])


def build_eval_transform(cfg):
    # images are 524x524 squares, so a plain resize keeps the whole object
    size = cfg["project"]["image_size"]
    norm = cfg["data"]["normalization"]
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(norm["mean"], norm["std"])])
