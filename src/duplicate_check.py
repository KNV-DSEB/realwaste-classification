"""Exact and near-duplicate detection on the frozen split (DATA_PROTOCOL duplicate policy).

Read-only: images and split files are never modified. The numbers produced here only
flag candidates; whether duplicates are "material" is decided by manual review and
recorded in DECISION_LOG.
"""
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import sha256_file


def dhash_bits(path, hash_size=16):
    """Difference hash: sign of horizontal gradients on a (hash_size+1)×hash_size grayscale thumbnail."""
    with Image.open(path) as img:
        thumb = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(thumb, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def compute_hashes(frame, hash_size=16):
    """SHA256 of file bytes and dHash bits for every row of `frame` (needs `image_path`)."""
    from tqdm.auto import tqdm

    sha256s, bits = [], []
    for path in tqdm(frame["image_path"], desc="hashing"):
        sha256s.append(sha256_file(path))
        bits.append(dhash_bits(path, hash_size))
    return sha256s, np.stack(bits)


def hamming_distances(bits_a, bits_b):
    """Pairwise Hamming distances between two sets of binary hashes."""
    a = bits_a.astype(np.float32)
    b = bits_b.astype(np.float32)
    distances = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - 2.0 * (a @ b.T)
    return np.rint(distances).astype(np.int32)


def _join_values(values):
    return ",".join(sorted(set(values)))


def exact_duplicate_groups(frame, sha256s):
    """Groups of byte-identical files, flagged when a group spans several splits or classes."""
    df = frame[["filepath", "class_name", "split"]].assign(sha256=sha256s)
    dup = df[df.duplicated("sha256", keep=False)]
    groups = dup.groupby("sha256").agg(
        size=("filepath", "size"),
        splits=("split", _join_values),
        classes=("class_name", _join_values),
        filepaths=("filepath", " | ".join),
    ).reset_index()
    groups["cross_split"] = groups["splits"].str.contains(",")
    groups["cross_class"] = groups["classes"].str.contains(",")
    return groups


def _pair_frame(frame, idx_a, idx_b, distances):
    a = frame.iloc[idx_a].reset_index(drop=True)
    b = frame.iloc[idx_b].reset_index(drop=True)
    pairs = pd.DataFrame({
        "distance": distances,
        "filepath_a": a["filepath"], "split_a": a["split"], "class_a": a["class_name"],
        "filepath_b": b["filepath"], "split_b": b["split"], "class_b": b["class_name"],
    })
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    pairs["cross_class"] = pairs["class_a"] != pairs["class_b"]
    return pairs.sort_values("distance", kind="stable").reset_index(drop=True)


def near_duplicate_pairs(frame, bits, max_distance):
    """All image pairs (any splits) whose dHash distance is <= max_distance."""
    distances = hamming_distances(bits, bits)
    idx_a, idx_b = np.nonzero(np.triu(distances <= max_distance, k=1))
    return _pair_frame(frame, idx_a, idx_b, distances[idx_a, idx_b])


def nearest_train_neighbors(frame, bits):
    """For every validation/test image, its closest training image by dHash distance.

    This is the leakage-relevant view: a small distance means a val/test image may have a
    near copy in the training data.
    """
    is_train = (frame["split"] == "train").to_numpy()
    train_idx = np.flatnonzero(is_train)
    other_idx = np.flatnonzero(~is_train)
    distances = hamming_distances(bits[other_idx], bits[train_idx])
    nearest = distances.argmin(axis=1)
    return _pair_frame(
        frame, other_idx, train_idx[nearest], distances[np.arange(len(other_idx)), nearest]
    )
