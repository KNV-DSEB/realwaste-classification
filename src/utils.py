import hashlib
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path=REPO_ROOT / "configs" / "config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(cfg):
    p = cfg["paths"]
    root = Path(p["drive_root"])
    return SimpleNamespace(
        drive_root=root,
        dataset_root=Path(p["dataset_root_override"] or root / p["dataset_dir"]),
        split_file=root / p["split_file"],
        class_mapping_file=root / p["class_mapping_file"],
        checkpoint_dir=root / p["checkpoint_dir"],
        result_dir=root / p["result_dir"],
        audit_dir=root / p["audit_dir"],
    )


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # each DataLoader worker gets its own numpy/random seed, derived from torch's
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def environment_info():
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit or None,
    }
