# E3 — EfficientNet-B0

Status: READY TO TRAIN (code tested locally on synthetic data; not yet run on the real data)

## Setup (DECISION_LOG D022)
- Model: `src/models/efficientnet.py` — torchvision EfficientNet-B0 with ImageNet weights `IMAGENET1K_V1`; head replaced by Dropout(0.2) → Linear(1280 → 9); 4,019,077 parameters.
- Stage A (head): backbone frozen, including the BatchNorm statistics of frozen blocks; 11,529 trainable parameters; AdamW lr 1e-3; max 10 epochs (`training.epochs_transfer_head`).
- Stage B (fine-tuning): starts from the best stage-A checkpoint; `features[6:]` (the three 7×7 blocks) + head trainable, 3,167,269 parameters; AdamW lr 1e-4; max 15 epochs (`training.epochs_finetune`).
- Both stages: weight decay 1e-4; ReduceLROnPlateau on validation macro F1 (factor 0.5, patience 2); early stopping on validation macro F1, patience 5 (D010); unweighted cross-entropy (D009); augmentation D011; ImageNet normalization (D008); batch 32; mixed precision on GPU.
- `E3_best.pt` is a copy of the better of `E3_stageA_best.pt` and `E3_stageB_best.pt` by validation macro F1 (ties keep stage A).
- Settings live in `configs/config.yaml` → `experiments.E3` and `training`.

## Run
`notebooks/07_train_E3.ipynb` on a Colab T4 GPU runtime → Run all. After a disconnect, Run all again: a finished stage is skipped and the interrupted one resumes. To start over, rename or delete `03_Checkpoints/E3/` on Drive.

## Outputs
- `03_Checkpoints/E3/E3_stageA_best.pt`, `E3_stageA_last.pt`, `E3_stageB_best.pt`, `E3_stageB_last.pt`, `E3_best.pt`
- `04_Results/experiments/E3/E3_history.csv` (columns `stage`, `stage_epoch`, global `epoch`), `E3_stageA_history.csv`, `E3_stageB_history.csv`, `E3_curves.png`, `E3_val_per_class_metrics.csv`, `E3_val_confusion_matrix.png`, `E3_training_summary.json`

## Interpretation rule
E3 differs from E1/E2 in pretrained weights, epoch budget and learning-rate schedule. Compare it as transfer learning vs training from scratch, not as an architecture comparison.

## Record here
- run ID and git commit (from `E3_training_summary.json`)
- per-stage best epoch and validation macro F1; selected stage
- issues and fixes
- final test results (only from the final-evaluation notebook)
