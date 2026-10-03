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
- **D015/D016 (current, verified 02b outputs):** grouped stratified split, blocks of 10 consecutive file numbers per class, seed 42 → `split_grouped_v2.csv`, SHA256 `65dc85e83acee56f5db0a0f3b9b5f08c84cf71d3d35c76f69c5213d4a33b3d3d`; train 3326 / val 719 / test 707. Same-class pairs 1 / 2 file numbers apart in different splits: 4.7 % / 9.4 % (D007: 46.7 % / 45.8 %).
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
- duplicate check run on Colab and reviewed (D014); grouped split approved (D015) and generated (D016); config switched to it
- pipeline smoke test (`04`, on the D007 config): all checks PASS; all 4752 images fully decode as RGB 524×524. Its "~0.8 min of data loading per epoch" was measured with a warm Drive cache (03 had just read every image); in a fresh Colab session the first epoch takes ~16 min (cold Drive reads, E1 epoch 1 = 945 s) and later epochs ~37 s
- duplicate check re-run on the grouped split: 0 exact duplicates; closest val/test↔train dHash distance 29; the 24 closest pairs are different objects; 1 of 32 spot-check triplets still has a re-shot across a block boundary (was 4) — **Gate 2 PASS (D017)**

- E1 code written (`src/models/simple_cnn.py`, `src/train.py`, `src/evaluate.py`, `notebooks/05_train_E1.ipynb`, D018)
- **E1 first run** (`E1_20261003-145155`, T4 GPU with AMP, 30 epochs, 33.5 min, commit `77b9c0f`). Validation only, best epoch 28: accuracy 0.652, macro-F1 0.643, macro precision 0.683, macro recall 0.639; 111,145 parameters. Train and validation curves stay close (no overfitting); still improving at the 30-epoch cap with LR already reduced to 1.25e-4 → extended to a 40-epoch cap (D019). Validation recall: Vegetation 0.97, Food Organics 0.82, Plastic 0.75, Cardboard 0.73, Paper 0.73, Metal 0.61, Glass 0.53, Miscellaneous Trash 0.33, Textile Trash 0.29; 113 of 250 errors are predictions of Plastic

- **E1 final** (same run extended under D019, early-stopped at epoch 38): best epoch 33 — validation accuracy 0.656, macro-F1 0.650, macro precision 0.706, macro recall 0.636; 54 % of validation errors are predictions of Plastic; recall Textile 0.29, Miscellaneous 0.30 — **Gate 3 PASS, E1 frozen (D021)**
- E2 code written (`src/models/multiscale_cnn.py`, `src/models/__init__.py` registry, `notebooks/06_train_E2.ipynb`, D020) and tested locally on synthetic data; not yet trained on the real data

- E3 code written (`src/models/efficientnet.py`, `notebooks/07_train_E3.ipynb`, D022) and tested locally on synthetic data, including loading the ImageNet weights; not yet trained on the real data
- **E2 final** (`E2_20261003-161916`, 40 epochs, no gain in the last 5): best epoch 35 — validation accuracy 0.787, macro-F1 0.787, macro precision 0.815, macro recall 0.772; mild overfitting after ~epoch 30 — **Gate 4 PASS, E2 frozen (D023)**
- class-weighting re-check after E2: Textile recall 0.29 → 0.51, Miscellaneous 0.30 → 0.70, errors predicted as Plastic 54 % → 33 % → **keep unweighted CE (D024)**

## Current gate
**Gate 5 — E3 EfficientNet-B0:** ready to train.

## Next actions
1. Run `notebooks/07_train_E3.ipynb`.
2. Final test evaluation of E1–E3 once, in one notebook, after all models are selected (D018).

## Open unknowns
- remaining same-object leakage at block boundaries: at most the 4.7 % / 9.4 % of neighbouring-number pairs that cross splits (most are different objects; 1 confirmed case in the 32-triplet spot check);
- same item photographed far apart in numbering (not covered by the spot check);
- each member's Drive mount path (set `paths.drive_root` in `configs/config.yaml` if it differs).

## Rule
Do not update this file with expected results. Only record verified project state.
