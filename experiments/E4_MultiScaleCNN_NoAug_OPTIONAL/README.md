# E4 — MultiScaleCNN_NoAug (optional ablation)

Status: READY TO TRAIN (code tested locally on synthetic data; not yet run on the real data)

## Question (RQ4)
What does training-time augmentation contribute? E4 repeats E2 with one factor changed.

## Setup (DECISION_LOG D025)
- Everything from E2 (`configs/config.yaml` → `experiments.E4.based_on: E2`): MultiScaleCNN architecture, AdamW lr 1e-3, weight decay 1e-4, ReduceLROnPlateau, max 40 epochs, early stopping on validation macro F1 (patience 5), unweighted cross-entropy, batch 32, AMP, seed 42 — so E4 starts from the same initial weights as E2.
- Only change: no training augmentation. Training images get the deterministic validation transform (resize to 224, ImageNet normalization).

## Run
`notebooks/08_train_E4.ipynb` on a Colab T4 GPU runtime → Run all. After a disconnect, Run all again: training resumes from `E4_last.pt`.

## Outputs
- `03_Checkpoints/E4/E4_best.pt`, `E4_last.pt`
- `04_Results/experiments/E4/E4_history.csv`, `E4_curves.png`, `E4_val_per_class_metrics.csv`, `E4_val_confusion_matrix.png`, `E4_training_summary.json`

## Interpretation rule
Compare E4 only with E2. A single run per setting cannot show whether a small difference is reliable; report it with the validation noise (about ±0.04 macro-F1 between epochs) in mind.

## Record here
- run ID and git commit
- best epoch, validation metrics, and how quickly validation loss diverges from training loss compared with E2
- final test results (only from the final-evaluation notebook)
