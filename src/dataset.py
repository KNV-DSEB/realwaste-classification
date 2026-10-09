import copy
import shutil
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_paths, seed_worker, sha256_file


def check_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"{path} has changed: its SHA256 no longer matches config.yaml")


def load_class_mapping(cfg):
    path = get_paths(cfg).class_mapping_file
    check_hash(path, cfg["split"]["class_mapping_sha256"])
    return pd.read_csv(path)


def load_frozen_split(cfg, check_files=True):
    paths = get_paths(cfg)
    check_hash(paths.split_file, cfg["split"]["split_sha256"])
    df = pd.read_csv(paths.split_file).merge(load_class_mapping(cfg), on="class_name", how="left")

    # filepath was recorded under the Drive folder used when the split was made;
    # image_path is where the same file is in this runtime
    old_root = PurePosixPath(cfg["paths"]["recorded_dataset_root"])
    df["image_path"] = [str(paths.dataset_root.joinpath(*PurePosixPath(f).relative_to(old_root).parts))
                        for f in df["filepath"]]
    if check_files:
        missing = [p for p in df["image_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images not found, e.g. {missing[0]}")
    return df


def use_local_copy(cfg, local_root="/content/RealWaste_local"):
    """Copy the images to the Colab disk once per session; reading them from Drive is slow."""
    local = Path(local_root)
    if not local.exists():
        tmp = local.with_name(local.name + "_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        print("Copying images to the local disk (~15 min, once per session)...")
        shutil.copytree(get_paths(cfg).dataset_root, tmp)
        tmp.rename(local)  # only a finished copy gets the final name
    cfg = copy.deepcopy(cfg)
    cfg["paths"]["dataset_root_override"] = str(local)
    return cfg


def verify_images_decode(df, expected_size=None):
    """Fully decode every image; returns the ones that fail."""
    bad = []
    for path in df["image_path"]:
        try:
            with Image.open(path) as img:
                img.load()
                if img.mode != "RGB" or (expected_size and img.size != tuple(expected_size)):
                    bad.append({"image_path": path, "error": f"{img.mode} {img.size}"})
        except Exception as e:
            bad.append({"image_path": path, "error": repr(e)})
    return pd.DataFrame(bad, columns=["image_path", "error"])


class RealWasteDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["class_index"].tolist()
        self.filepaths = df["filepath"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[i], self.filepaths[i]


def make_loader(df, transform, cfg, shuffle):
    workers = cfg["data"]["num_workers"]
    return DataLoader(
        RealWasteDataset(df, transform),
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(cfg["project"]["seed"]),
    )


def get_dataloaders(cfg, augment=True, split_df=None):
    """Train and validation loaders. augment=False is the E4 ablation."""
    df = load_frozen_split(cfg) if split_df is None else split_df
    train_tf = build_train_transform(cfg) if augment else build_eval_transform(cfg)
    return (make_loader(df[df["split"] == "train"], train_tf, cfg, shuffle=True),
            make_loader(df[df["split"] == "val"], build_eval_transform(cfg), cfg, shuffle=False))


def get_test_loader(cfg, split_df=None):
    """Only for the final evaluation (notebook 09)."""
    df = load_frozen_split(cfg) if split_df is None else split_df
    return make_loader(df[df["split"] == "test"], build_eval_transform(cfg), cfg, shuffle=False)
