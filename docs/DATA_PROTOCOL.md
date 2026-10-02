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

Required audit artifacts (in `04_Results/dataset_audit/` on Drive unless noted):
- `dataset_summary.csv` — `03_duplicate_check.ipynb`
- `class_distribution.csv`, `folder_image_counts.csv`, `image_metadata.csv`, `corrupted_images.csv`, `sample_grid.png` — `01_dataset_audit.ipynb`
- duplicate reports (`exact_duplicate_groups.csv`, `near_duplicate_pairs.csv`, `nearest_train_neighbors.csv`, `nearest_train_distance_hist.png`, `duplicate_review_grid.png`) — `03_duplicate_check.ipynb`
- short audit notes — kept in the repo as `report/dataset_audit_notes.md`

## Gate 2 — Split
Only after Gate 1 PASS:
- use only valid images;
- create stratified 70/15/15 split with seed 42;
- save one immutable `split.csv`;
- verify class distribution in all splits;
- record exact counts;
- never independently random-split inside model notebooks.

Frozen split.csv columns (D007):
`filepath, class_name, width, height, mode, extension, split`

Class indices come from `class_mapping.csv` (`class_name, class_index`), joined on `class_name` by `src/dataset.py`. `filepath` is the row ID, recorded under the Drive root used at split time (`paths.recorded_dataset_root`); `src/dataset.py` maps it to the image location of the current runtime. Never edit or re-save `split.csv` (its SHA256 is checked).

## Preprocessing rule
Fit/decide any train-dependent preprocessing using training data only. Random augmentation must be training-only.

## Duplicate policy
If near-duplicate sequences exist, do not let visually near-identical copies leak across train/test. Escalate to ChatGPT before freezing split if this risk appears material.

Check: `notebooks/03_duplicate_check.ipynb` (exact SHA256 duplicates + 256-bit dHash near-duplicates, run read-only against the frozen split). Record the outcome in DECISION_LOG before any training.
