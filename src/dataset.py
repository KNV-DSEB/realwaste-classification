"""Shared RealWaste data pipeline: frozen split -> Dataset -> DataLoaders.

Every experiment (E1-E4) must load data through this module so the split, class mapping,
transforms and loader settings are identical (CONSTITUTION C2, EXPERIMENT_PROTOCOL).
"""
import copy
import shutil
from pathlib import Path, PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.transforms import build_eval_transform, build_train_transform
from src.utils import get_paths, make_generator, seed_worker, sha256_file

SPLIT_COLUMNS = {"filepath", "class_name", "split"}


def _check_sha256(path, expected, name):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{name} SHA256 mismatch at {path}: expected {expected}, got {actual}. "
            "The frozen artifact has changed; stop and check DECISION_LOG D007."
        )


def load_class_mapping(cfg):
    """Frozen class mapping (class_name -> class_index), verified by SHA256."""
    paths = get_paths(cfg)
    _check_sha256(paths.class_mapping_file, cfg["split"]["class_mapping_sha256"], "class_mapping.csv")
    mapping = pd.read_csv(paths.class_mapping_file)
    num_classes = cfg["split"]["num_classes"]
    if list(mapping.columns) != ["class_name", "class_index"]:
        raise ValueError(f"Unexpected class_mapping.csv columns: {list(mapping.columns)}")
    if mapping["class_index"].tolist() != list(range(num_classes)) or not mapping["class_name"].is_unique:
        raise ValueError("class_mapping.csv must list each class once with indices 0..K-1 in order")
    return mapping


def load_frozen_split(cfg, check_files=True):
    """Verified frozen split with added `class_index` and `image_path` columns.

    `filepath` is kept unchanged as the stable row ID. It was recorded under the Drive root
    used at split time (`paths.recorded_dataset_root`); `image_path` is where the same image
    lives in this runtime (`paths.drive_root` or `paths.dataset_root_override`).
    """
    paths = get_paths(cfg)
    split_cfg = cfg["split"]
    _check_sha256(paths.split_file, split_cfg["split_sha256"], "split.csv")
    mapping = load_class_mapping(cfg)
    df = pd.read_csv(paths.split_file)

    missing_columns = SPLIT_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"split.csv is missing columns: {sorted(missing_columns)}")
    if not df["filepath"].is_unique:
        raise ValueError("split.csv contains duplicate filepaths")
    counts = df["split"].value_counts().to_dict()
    if counts != split_cfg["expected_counts"]:
        raise ValueError(f"Split counts {counts} != expected {split_cfg['expected_counts']}")
    if set(df["class_name"]) != set(mapping["class_name"]):
        raise ValueError("Class names in split.csv and class_mapping.csv differ")

    df = df.merge(mapping, on="class_name", how="left", validate="many_to_one")

    recorded_root = PurePosixPath(cfg["paths"]["recorded_dataset_root"])
    df["image_path"] = [
        str(paths.dataset_root.joinpath(*PurePosixPath(fp).relative_to(recorded_root).parts))
        for fp in df["filepath"]
    ]
    if check_files:
        missing = [p for p in df["image_path"] if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} images not found under {paths.dataset_root}, e.g. {missing[:3]}. "
                "Check paths.drive_root / paths.dataset_root_override in config.yaml."
            )
    return df


def use_local_copy(cfg, local_root="/content/RealWaste_local"):
    """Return a config whose images are read from a copy on the runtime's local disk.

    The first read of the images from Drive is slow (~15 min); copying them once per Colab session lets
    several trials share that cost. Split files are still read (and SHA256-checked) from Drive. An
    interrupted copy is discarded and redone.
    """
    local = Path(local_root)
    if not local.exists():
        partial = local.with_name(local.name + "_partial")
        shutil.rmtree(partial, ignore_errors=True)
        print(f"Copying images from {get_paths(cfg).dataset_root} to {local} (once per session)...")
        shutil.copytree(get_paths(cfg).dataset_root, partial)
        partial.rename(local)
    local_cfg = copy.deepcopy(cfg)
    local_cfg["paths"]["dataset_root_override"] = str(local)
    return local_cfg


def verify_images_decode(frame, expected_size=None):
    """Fully decode every image (`Image.load`), unlike the header-only `verify()` in the audit.

    Returns a DataFrame of failures; empty means every image decoded as RGB of the expected size.
    """
    from tqdm.auto import tqdm

    failures = []
    for path in tqdm(frame["image_path"], desc="decoding"):
        try:
            with Image.open(path) as img:
                img.load()
                if img.mode != "RGB" or (expected_size and img.size != tuple(expected_size)):
                    failures.append({"image_path": path, "error": f"mode={img.mode} size={img.size}"})
        except Exception as e:
            failures.append({"image_path": path, "error": repr(e)})
    return pd.DataFrame(failures, columns=["image_path", "error"])


class RealWasteDataset(Dataset):
    """Yields (image_tensor, class_index, filepath); `filepath` is the split.csv row ID."""

    def __init__(self, frame, transform):
        self.image_paths = frame["image_path"].tolist()
        self.labels = frame["class_index"].astype("int64").tolist()
        self.filepaths = frame["filepath"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as img:
            image = img.convert("RGB")
        return self.transform(image), self.labels[index], self.filepaths[index]


def _make_loader(dataset, cfg, shuffle):
    num_workers = cfg["data"]["num_workers"]
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg["data"]["pin_memory"] and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["project"]["seed"]),
    )


def get_dataloaders(cfg, augment=True, split_df=None):
    """Train and validation loaders only.

    The test loader is deliberately not built here: use `get_test_loader` once per
    validation-selected checkpoint for the final evaluation (CONSTITUTION C3).
    `augment=False` is the E4 ablation: training images get the deterministic transform.
    """
    if split_df is None:
        split_df = load_frozen_split(cfg)
    train_transform = build_train_transform(cfg) if augment else build_eval_transform(cfg)
    train_set = RealWasteDataset(split_df[split_df["split"] == "train"], train_transform)
    val_set = RealWasteDataset(split_df[split_df["split"] == "val"], build_eval_transform(cfg))
    return _make_loader(train_set, cfg, shuffle=True), _make_loader(val_set, cfg, shuffle=False)


def get_test_loader(cfg, split_df=None):
    """FINAL EVALUATION ONLY. Never use for model selection, tuning or early stopping."""
    if split_df is None:
        split_df = load_frozen_split(cfg)
    test_set = RealWasteDataset(split_df[split_df["split"] == "test"], build_eval_transform(cfg))
    return _make_loader(test_set, cfg, shuffle=False)
