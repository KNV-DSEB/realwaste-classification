"""Duplicate check (notebook 03): SHA256 finds byte-identical files, a difference hash (dHash)
finds near copies. The numbers only flag candidates; the final call is made by looking at the images."""
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import sha256_file


def dhash_bits(path, hash_size=16):
    # shrink to a tiny grayscale image and record whether each pixel is brighter than its right neighbour
    with Image.open(path) as img:
        px = np.asarray(img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.int16)
    return (px[:, 1:] > px[:, :-1]).flatten()


def compute_hashes(frame, hash_size=16):
    sha = [sha256_file(p) for p in frame["image_path"]]
    bits = np.stack([dhash_bits(p, hash_size) for p in frame["image_path"]])
    return sha, bits


def hamming_distances(a, b):
    """Number of differing bits between every row of a and every row of b."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    return np.rint(a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * a @ b.T).astype(np.int32)


def exact_duplicate_groups(frame, sha256s):
    df = frame[["filepath", "class_name", "split"]].assign(sha256=sha256s)
    dup = df[df.duplicated("sha256", keep=False)]
    join = lambda s: ",".join(sorted(set(s)))
    groups = dup.groupby("sha256").agg(size=("filepath", "size"), splits=("split", join),
                                       classes=("class_name", join), filepaths=("filepath", " | ".join)).reset_index()
    groups["cross_split"] = groups["splits"].str.contains(",")
    groups["cross_class"] = groups["classes"].str.contains(",")
    return groups


def _pairs(frame, ia, ib, dist):
    a, b = frame.iloc[ia].reset_index(drop=True), frame.iloc[ib].reset_index(drop=True)
    pairs = pd.DataFrame({"distance": dist, "filepath_a": a["filepath"], "split_a": a["split"], "class_a": a["class_name"],
                          "filepath_b": b["filepath"], "split_b": b["split"], "class_b": b["class_name"]})
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    pairs["cross_class"] = pairs["class_a"] != pairs["class_b"]
    return pairs.sort_values("distance", kind="stable").reset_index(drop=True)


def near_duplicate_pairs(frame, bits, max_distance):
    d = hamming_distances(bits, bits)
    ia, ib = np.nonzero(np.triu(d <= max_distance, k=1))
    return _pairs(frame, ia, ib, d[ia, ib])


def nearest_train_neighbors(frame, bits):
    """For each val/test image, the closest training image. A small distance hints at leakage."""
    is_train = (frame["split"] == "train").to_numpy()
    train, other = np.flatnonzero(is_train), np.flatnonzero(~is_train)
    d = hamming_distances(bits[other], bits[train])
    nearest = d.argmin(1)
    return _pairs(frame, other, train[nearest], d[np.arange(len(other)), nearest])
