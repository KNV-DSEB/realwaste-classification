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
Checked by `notebooks/03_duplicate_check.ipynb` on the D007 split (DECISION_LOG D014):
- exact duplicates (SHA256): 0 groups.
- near duplicates (256-bit dHash ≤ 20): 1 pair, two Food Organics images both in train (d = 10).
- nearest training image for each of the 1426 val/test images: distance 22–104 (median 80).
- manual review of the 24 closest val/test ↔ train pairs: 1 near-identical photo (test Metal_216 ↔ train Metal_217, d = 22), 1 re-shot of the same milk carton (val Cardboard_142 ↔ train Cardboard_140, d = 43), 22 different objects on a similar background.
- spot check of 32 random triplets of consecutive file numbers: 17/32 show the same item photographed again, usually 1–2 numbers apart (often moved or flipped, so hashing misses it); 4/32 had that item in train and in val/test.

Conclusion: re-shots of the same item are systematic. The image-level split D007 is replaced by a split grouped into blocks of 10 consecutive file numbers per class (D015, generated as D016: train 3326 / val 719 / test 707). Same-class images 1 or 2 file numbers apart now fall into different splits in 4.7 % / 9.4 % of pairs, down from 46.7 % / 45.8 %, and all six same-object pairs found above now share a split. Remaining limitation: re-shots that straddle a block boundary, or the same item photographed far apart in numbering, can still cross splits.

Re-check on the grouped split (D017): 0 exact duplicates; the closest val/test ↔ train dHash distance rose from 22 to 29; the 24 closest pairs are all different objects; in the same 32 spot-check triplets, re-shots crossing splits fell from 4 to 1 (Miscellaneous Trash 59 val ↔ 60 train, a heart ornament photographed front and back across a block boundary).

## Gate recommendation
Gate 1: PASS (facts verified; duplicate risk handled by the grouped split). Gate 2: PASS on `split_grouped_v2.csv` (D017). Report the residual boundary leakage as a limitation.
