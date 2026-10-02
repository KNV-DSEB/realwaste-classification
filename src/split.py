"""Grouped stratified split (DECISION_LOG D015).

RealWaste photographs many items 2-3 times under neighbouring file numbers of the same class
(duplicate check: 17/32 random triplets of consecutive files contained the same object).
An image-level random split therefore puts copies of one item into both train and val/test.
Here, consecutive file numbers within a class are grouped into fixed blocks
(`number // block_size`) and every block goes to exactly one split.
"""
import numpy as np
import pandas as pd

BASE_COLUMNS = ["filepath", "class_name", "width", "height", "mode", "extension"]


def file_number(filepaths):
    """Trailing number of each filename, e.g. '.../Plastic_646.jpg' -> 646."""
    numbers = filepaths.str.extract(r"_(\d+)\.[A-Za-z]+$", expand=False)
    if numbers.isna().any():
        raise ValueError(f"Filenames without a trailing number: {filepaths[numbers.isna()].head().tolist()}")
    return numbers.astype(int)


def grouped_stratified_split(frame, block_size, val_ratio, test_ratio, seed):
    """Assign whole blocks of consecutive file numbers to train/val/test, class by class.

    Per class, blocks are shuffled with one seeded generator (classes in sorted order); the first
    round(test_ratio * n_blocks) blocks go to test, the next round(val_ratio * n_blocks) to val,
    the rest to train. The result does not depend on the row order of `frame`.
    """
    df = frame[BASE_COLUMNS].sort_values("filepath").reset_index(drop=True)
    df["block"] = file_number(df["filepath"]) // block_size
    rng = np.random.default_rng(seed)
    assignment = {}
    for class_name in sorted(df["class_name"].unique()):
        blocks = np.sort(df.loc[df["class_name"] == class_name, "block"].unique())
        n_test = round(test_ratio * len(blocks))
        n_val = round(val_ratio * len(blocks))
        for position, block in enumerate(rng.permutation(blocks)):
            if position < n_test:
                split = "test"
            elif position < n_test + n_val:
                split = "val"
            else:
                split = "train"
            assignment[(class_name, block)] = split
    df["split"] = [assignment[key] for key in zip(df["class_name"], df["block"])]
    return df.sort_values(["split", "class_name", "filepath"]).reset_index(drop=True)


def adjacent_pair_summary(frame, max_gap=2):
    """How often images of the same class whose file numbers differ by 1..max_gap fall into
    different splits. This upper-bounds the remaining same-object leakage."""
    df = frame[["class_name", "split"]].assign(num=file_number(frame["filepath"]).to_numpy())
    pairs = pd.concat(
        [
            df.merge(df.assign(num=df["num"] - gap), on=["class_name", "num"], suffixes=("_a", "_b")).assign(gap=gap)
            for gap in range(1, max_gap + 1)
        ],
        ignore_index=True,
    )
    pairs["cross_split"] = pairs["split_a"] != pairs["split_b"]
    summary = pairs.groupby("gap")["cross_split"].agg(pairs="size", cross_split="sum")
    summary["cross_split_rate"] = (summary["cross_split"] / summary["pairs"]).round(3)
    return summary
