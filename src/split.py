"""Grouped split (D015): the same item is often photographed 2-3 times under neighbouring file numbers,
so images are grouped into blocks of consecutive numbers and each block goes to one split."""
import numpy as np
import pandas as pd

BASE_COLUMNS = ["filepath", "class_name", "width", "height", "mode", "extension"]


def file_number(filepaths):
    """'.../Plastic_646.jpg' -> 646"""
    return filepaths.str.extract(r"_(\d+)\.[A-Za-z]+$", expand=False).astype(int)


def grouped_stratified_split(frame, block_size, val_ratio, test_ratio, seed):
    df = frame[BASE_COLUMNS].sort_values("filepath").reset_index(drop=True)
    df["block"] = file_number(df["filepath"]) // block_size
    rng = np.random.default_rng(seed)
    split_of = {}
    for cls in sorted(df["class_name"].unique()):
        blocks = rng.permutation(np.sort(df.loc[df["class_name"] == cls, "block"].unique()))
        n_test, n_val = round(test_ratio * len(blocks)), round(val_ratio * len(blocks))
        for i, block in enumerate(blocks):
            split_of[cls, block] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    df["split"] = [split_of[key] for key in zip(df["class_name"], df["block"])]
    return df.sort_values(["split", "class_name", "filepath"]).reset_index(drop=True)


def adjacent_pair_summary(frame, max_gap=2):
    """Share of same-class images 1..max_gap file numbers apart that ended up in different splits."""
    df = frame[["class_name", "split"]].assign(num=file_number(frame["filepath"]).to_numpy())
    pairs = pd.concat([df.merge(df.assign(num=df["num"] - gap), on=["class_name", "num"]).assign(gap=gap)
                       for gap in range(1, max_gap + 1)])
    pairs["cross_split"] = pairs["split_x"] != pairs["split_y"]
    summary = pairs.groupby("gap")["cross_split"].agg(pairs="size", cross_split="sum")
    summary["cross_split_rate"] = (summary["cross_split"] / summary["pairs"]).round(3)
    return summary
