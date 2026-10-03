# E2 — MultiScaleCNN

Status: READY TO TRAIN (code tested locally on synthetic data; not yet run on the real data)

## Setup (DECISION_LOG D020)
- Model: `src/models/multiscale_cnn.py`
  - stem: Conv3×3 (32) → BN → ReLU → MaxPool
  - 4 multi-scale blocks (widths 64, 128, 256, 256): parallel 1×1 / 3×3 / 5×5 convolutions (1/4, 1/2, 1/4 of the channels) → concatenate → BN → ReLU → MaxPool
  - head: GAP → Dense(128) → ReLU → Dropout(0.3) → Dense(9)
  - 1,230,377 parameters (E1: 111,145); receptive field of the 5×5 path 154 px (E1: 22 px)
- Training identical to E1: AdamW lr 1e-3, weight decay 1e-4; ReduceLROnPlateau on validation macro F1 (factor 0.5, patience 2); unweighted cross-entropy (D009); augmentation D011; batch 32; max 40 epochs (D019); early stopping and selection on validation macro F1, patience 5 (D010); mixed precision on GPU.
- Settings live in `configs/config.yaml` → `experiments.E2` and `training`.

## Run
`notebooks/06_train_E2.ipynb` on a Colab T4 GPU runtime → Run all. After a disconnect, Run all again: training resumes from `E2_last.pt`. To start over, rename or delete `03_Checkpoints/E2/` on Drive.

## Outputs
- `03_Checkpoints/E2/E2_best.pt`, `E2_last.pt`
- `04_Results/experiments/E2/E2_history.csv`, `E2_curves.png`, `E2_val_per_class_metrics.csv`, `E2_val_confusion_matrix.png`, `E2_training_summary.json`

## Interpretation rule
E2 differs from E1 in depth, width and multi-scale branches at the same time. Describe a difference as "the deeper multi-scale model", not as an effect of multi-scale alone.

## Record here
- run IDs and git commit (from `E2_training_summary.json`)
- best epoch and validation metrics
- issues and fixes
- final test results (only from the final-evaluation notebook)
