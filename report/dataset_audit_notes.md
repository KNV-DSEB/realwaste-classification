# Dataset Audit Notes

Sources: outputs of `notebooks/01_dataset_audit.ipynb` and `notebooks/02_split_generator.ipynb` (run 2026-10-02). Sections marked *(draft)* were written from the 27-image sample grid by Claude and need a team member's confirmation.

## Root path
`/content/drive/MyDrive/Deep Learning - RealWaste/01_Dataset/RealWaste`
(Kaggle `joebeachcapital/realwaste` version 1, folder `realwaste-main/RealWaste`)

## Class folders
9 folders, no extras: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation.

## Total valid images
4752 (4752 image files found).

## Corrupted images
0, checked with PIL `verify()` (header level). A full-decode pass over every image runs in `notebooks/04_pipeline_smoke_test.ipynb`.

## Modes / dimensions
All RGB, all `.jpg`, all 524 × 524.

## Imbalance
Largest Plastic 921, smallest Textile Trash 318; ratio 2.90 (moderate). Handled with macro metrics and an unweighted CE baseline (D009). Textile Trash has only 48 test images, so per-class differences of a few points are within noise.

## Visual observations *(draft)*
- One capture setup: a single item, photographed top-down on a concrete floor.
- The background is shared across classes; lighting and colour cast vary between images (some grey, some bluish).
- One Cardboard sample is printed food packaging that could also be read as Paper or Plastic; label conventions for mixed-material items should be noted.
- Team: add visually confusing class pairs and occlusion notes.

## Duplicate / near-duplicate risk
Not checked by 01. Run `notebooks/03_duplicate_check.ipynb` and record here:
- exact-duplicate groups (cross-split / cross-class):
- near-duplicate candidates (dHash ≤ 20) across splits:
- manual review of the closest val/test ↔ train pairs:
- limitation: the same object photographed from a different angle cannot be detected by hashing.

## Gate recommendation
FIX BEFORE CONTINUING until the duplicate check is recorded in DECISION_LOG; then PASS or FIX according to that result.
