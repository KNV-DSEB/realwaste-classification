# E1 — SimpleCNN

Status: TRAINED to 30 epochs; extension to the 40-epoch cap pending (D019)

## Runs
| Run ID | Commit | Epochs | Best epoch | Val accuracy | Val macro-F1 | Val macro P / R | Time | Notes |
|---|---|---|---|---|---|---|---|---|
| E1_20261003-145155 | `df98d0a` (code), outputs `77b9c0f` | 30 / 30 (cap) | 28 | 0.652 | 0.643 | 0.683 / 0.639 | 33.5 min (epoch 1: 945 s cold Drive read) | T4 + AMP; LR 1e-3 → 1.25e-4 by plateau steps at epochs 8, 21, 26; no overfitting; still improving at the cap → extend to 40 |

Validation observations (best epoch): Plastic acts as a sink (218 predicted, 105 correct; 113 of 250 errors); lowest recall Textile Trash 0.29 (→ Plastic 12, Paper 11, Metal 7) and Miscellaneous Trash 0.33 (→ Plastic 31); Glass → Plastic 20.

## Setup (DECISION_LOG D018)
- Model: `src/models/simple_cnn.py` — 3 × [Conv3×3 (32/64/128) → BN → ReLU → MaxPool] → GAP → Dense(128) → ReLU → Dropout(0.3) → Dense(9).
- Data: frozen grouped split `split_grouped_v2.csv` (D016), ImageNet normalization (D008), training augmentation D011.
- Loss: unweighted cross-entropy (D009). Optimizer: AdamW, lr 1e-3, weight decay 1e-4. Scheduler: ReduceLROnPlateau on validation macro F1 (factor 0.5, patience 2).
- Max 40 epochs (D019; was 30 in D018); batch 32; early stopping and checkpoint selection on validation macro F1, patience 5 (D010); mixed precision on GPU.
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
