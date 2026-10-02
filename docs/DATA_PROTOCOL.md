# DATA_PROTOCOL

## Gate 1 — Dataset Audit
Before splitting, verify:
- root directory exists;
- class folder names;
- number of images per class;
- total readable images;
- corrupt/unreadable images;
- extension distribution;
- RGB/grayscale/other mode distribution;
- image dimension distribution;
- random visual samples per class;
- obvious duplicate/near-duplicate patterns;
- imbalance ratio.

Required audit artifacts:
- `dataset_summary.csv`
- `class_distribution.csv`
- `image_metadata.csv`
- `corrupted_images.csv`
- `sample_grid.png`
- short `audit_notes.md`

## Gate 2 — Split
Only after Gate 1 PASS:
- use only valid images;
- create stratified 70/15/15 split with seed 42;
- save one immutable `split.csv`;
- verify class distribution in all splits;
- record exact counts;
- never independently random-split inside model notebooks.

Suggested split.csv columns:
`filepath, class_name, class_id, split`

## Preprocessing rule
Fit/decide any train-dependent preprocessing using training data only. Random augmentation must be training-only.

## Duplicate policy
If near-duplicate sequences exist, do not let visually near-identical copies leak across train/test. Escalate to ChatGPT before freezing split if this risk appears material.
