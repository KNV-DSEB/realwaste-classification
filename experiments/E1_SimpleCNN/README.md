# E1 — SimpleCNN

Status: READY TO TRAIN (code tested locally on synthetic data; not yet run on the real data)

## Setup (DECISION_LOG D018)
- Model: `src/models/simple_cnn.py` — 3 × [Conv3×3 (32/64/128) → BN → ReLU → MaxPool] → GAP → Dense(128) → ReLU → Dropout(0.3) → Dense(9).
- Data: frozen grouped split `split_grouped_v2.csv` (D016), ImageNet normalization (D008), training augmentation D011.
- Loss: unweighted cross-entropy (D009). Optimizer: AdamW, lr 1e-3, weight decay 1e-4. Scheduler: ReduceLROnPlateau on validation macro F1 (factor 0.5, patience 2).
- Max 30 epochs; batch 32; early stopping and checkpoint selection on validation macro F1, patience 5 (D010); mixed precision on GPU.
- Settings live in `configs/config.yaml` → `experiments.E1` and `training`.

## Run
`notebooks/05_train_E1.ipynb` on a Colab T4 GPU runtime → Run all. After a disconnect, Run all again: training resumes from `E1_last.pt`. To start over, rename or delete `03_Checkpoints/E1/` on Drive.

## Outputs
- `03_Checkpoints/E1/E1_best.pt`, `E1_last.pt`
- `04_Results/experiments/E1/E1_history.csv`, `E1_curves.png`, `E1_val_per_class_metrics.csv`, `E1_val_confusion_matrix.png`, `E1_training_summary.json`

## Record here
- run IDs and git commit (from `E1_training_summary.json`)
- best epoch and validation metrics
- issues and fixes
- final test results (only from the final-evaluation notebook)
