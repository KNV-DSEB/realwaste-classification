"""Shared helpers: config, project paths, file hashing, seeding, environment record."""
import hashlib
import platform
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ProjectPaths:
    drive_root: Path
    dataset_root: Path
    split_file: Path
    class_mapping_file: Path
    checkpoint_dir: Path
    result_dir: Path
    audit_dir: Path


def get_paths(cfg):
    """Absolute paths for this runtime, all derived from `paths.drive_root`.

    `paths.dataset_root_override` (e.g. a local /content copy) replaces only the image root;
    the frozen split files are always read from the shared Drive folder.
    """
    p = cfg["paths"]
    drive_root = Path(p["drive_root"])
    override = p.get("dataset_root_override")
    return ProjectPaths(
        drive_root=drive_root,
        dataset_root=Path(override) if override else drive_root / p["dataset_dir"],
        split_file=drive_root / p["split_file"],
        class_mapping_file=drive_root / p["class_mapping_file"],
        checkpoint_dir=drive_root / p["checkpoint_dir"],
        result_dir=drive_root / p["result_dir"],
        audit_dir=drive_root / p["audit_dir"],
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy and PyTorch. Results are repeatable in practice but not
    bit-exact on GPU unless `deterministic=True` (slower cuDNN kernels)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader `worker_init_fn`: derive each worker's NumPy/random seed from torch."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _git(*args):
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def environment_info():
    """Versions, hardware and git state to store in each run's config_snapshot (CONSTITUTION C9)."""
    import pandas
    import PIL
    import sklearn
    import torchvision

    status = _git("status", "--porcelain")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }
