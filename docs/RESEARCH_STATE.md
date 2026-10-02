# RESEARCH_STATE

**Status date:** 2026-10-02

## Problem
Multi-class real-world waste image classification.

## Dataset (verified: `01_dataset_audit.ipynb` outputs, D006)
- RealWaste from Kaggle (`joebeachcapital/realwaste`, version 1), copied to `01_Dataset/RealWaste/` on the shared Drive.
- 4752 images, 9 classes; all RGB, `.jpg`, 524 × 524; 0 unreadable (PIL `verify()`, header level).
- Per class: Cardboard 461, Food Organics 411, Glass 420, Metal 790, Miscellaneous Trash 495, Paper 500, Plastic 921, Textile Trash 318, Vegetation 436. Largest/smallest = 2.90.

## Split
- **D007 (superseded):** image-level stratified 70/15/15, seed 42: train 3326 / val 713 / test 713; `split.csv` SHA256 `043e68e8028e402f2c2d04cdfa48e44b54c9ac6099981387f2859a0724aa4b1b`. Kept on Drive as a record.
- **D014 (verified, `03_duplicate_check.ipynb`):** 0 exact duplicates, but the same item is often photographed 2–3 times under neighbouring file numbers (17/32 spot-check triplets); under D007 such re-shots sit in train and val/test.
- **D015 (approved, not yet generated):** grouped stratified split, blocks of 10 consecutive file numbers per class, seed 42 → `split_grouped_v2.csv` via `02b_grouped_split.ipynb`. Hash and counts to be recorded after the Colab run.
- `class_mapping.csv` SHA256 `a99922d406b2a308ff6e85e0015dc675ad24be07f7e481ff4f0a1ed65a8aed8d`
- Class mapping: 0 Cardboard, 1 Food Organics, 2 Glass, 3 Metal, 4 Miscellaneous Trash, 5 Paper, 6 Plastic, 7 Textile Trash, 8 Vegetation.

## Locked design
- E1 SimpleCNN
- E2 MultiScaleCNN
- E3 EfficientNet-B0
- E4 optional augmentation ablation
- 224×224
- stratified 70/15/15 split, grouped by blocks of consecutive file numbers (D015)
- seed 42
- ImageNet normalization for all models (D008); unweighted CE (D009); selection and early stopping on validation Macro F1 (D010); shared augmentation profile (D011)

## Completed
- scope selected; architecture families selected; 10-day plan defined; starter pack initialized
- Colab + Drive + GitHub setup; dataset downloaded and copied to Drive
- dataset audit (counts, modes, extensions, dimensions, readability, class distribution, sample grid)
- split generated and frozen with SHA256 fingerprints (D007)
- pre-training audit; protocol decisions D006–D013 recorded
- shared data pipeline written (`src/utils.py`, `src/transforms.py`, `src/dataset.py`)
- duplicate check run on Colab and reviewed (D014); grouped split approved (D015); `src/split.py` and `02b_grouped_split.ipynb` written and tested locally on a synthetic 4752-row file list

## Current gate
**Gate 2 — Split: being redone (D015).** Gate 1 audit facts are verified; its duplicate requirement is satisfied once the grouped split is generated and checked.

## Next actions
1. Run `notebooks/02b_grouped_split.ipynb`; record the printed SHA256 and counts in `configs/config.yaml` (switch `split_file` to `split_grouped_v2.csv`).
2. Re-run `notebooks/03_duplicate_check.ipynb` and `notebooks/04_pipeline_smoke_test.ipynb` on the new split.
3. Gate 3: implement and train E1 SimpleCNN through the shared pipeline.

## Open unknowns
- remaining same-object leakage at block boundaries (measured by 02b);
- same item photographed far apart in numbering (not covered by the spot check);
- each member's Drive mount path (set `paths.drive_root` in `configs/config.yaml` if it differs).

## Rule
Do not update this file with expected results. Only record verified project state.
