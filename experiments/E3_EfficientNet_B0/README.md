# E3 — EfficientNet-B0

Status: DONE — frozen (D027). `E3_best.pt` = stage B, stage epoch 15 (global epoch 25).

## Runs
| Run ID | Code | Stage A (head) | Stage B (fine-tune) | Selected | Val accuracy | Val macro-F1 | Val macro P / R | Time |
|---|---|---|---|---|---|---|---|---|
| E3_20261003-171100 | `5b26ae8`, outputs `5e8b363` | 10 / 10 epochs, best epoch 10, val macro-F1 0.810 | 15 / 15 epochs, best stage epoch 15, val macro-F1 0.879 (stage epoch 10: 0.879 as well; LR 1e-4 → 5e-5 at stage epoch 14) | **B** (global epoch 25) | **0.880** | **0.879** | 0.894 / 0.871 | 26.5 min (stage A 16.5, of which epoch 1 = 639 s cold Drive read; stage B 10.1, ~40 s/epoch) |

Training behaviour:
- Neither stage early-stopped; both ended at their epoch cap. In stage B validation macro-F1 was flat over the last six epochs (about 0.868–0.879) while training accuracy approached 1.0 (training loss ~0.02 vs validation loss ~0.42): the fine-tuned network memorises the training set and further epochs would not help generalisation, so E3 was not extended (D027).
- The bump in training loss/accuracy at the start of stage B (epoch 11) is expected: the newly unfrozen blocks switch their BatchNorm layers to batch statistics and their stochastic depth becomes active; validation metrics were not affected.

Validation observations at the selected checkpoint (719 images): correct 633. Recall: Food Organics 0.97, Plastic 0.94, Vegetation 0.94, Paper 0.93, Metal 0.92, Cardboard 0.84, Textile Trash 0.82, Glass 0.77, Miscellaneous Trash 0.71. Largest confusions: Miscellaneous → Plastic 9, Glass → Plastic 8, Metal → Plastic 6, Cardboard → Metal 5, Miscellaneous → Metal 5. Errors predicted as Plastic: 27 of 86 (31 %).

Compared on validation (selected checkpoints): macro-F1 E1 0.650 → E2 0.787 → E3 0.879; accuracy 0.656 → 0.787 → 0.880.

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
