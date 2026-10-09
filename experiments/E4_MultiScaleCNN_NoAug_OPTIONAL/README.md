# E4 — MultiScaleCNN_NoAug (optional ablation)

Status: DONE — frozen (D028). Selected checkpoint `E4_best.pt` = epoch 29.

## Runs
| Run ID | Code | Epochs | Best epoch | Val accuracy | Val macro-F1 | Val macro P / R | Time |
|---|---|---|---|---|---|---|---|
| E4_20261004-033943 | `5e8b363`, outputs `6ab11ea` | 34 / 40, early-stopped | **29** | **0.814** | **0.808** | 0.819 / 0.803 | 64.4 min in total: the session was interrupted after epoch 27 and resumed at epoch 28, so the cold Drive read (~16–17 min) happened twice; later epochs ~33 s |

## Result vs E2 (validation, selected checkpoints)
| | E2 (augmentation D011) | E4 (no augmentation) |
|---|---|---|
| Accuracy | 0.787 | 0.814 |
| Macro-F1 | 0.787 | 0.808 |
| Errors predicted as Plastic | 51 of 153 | 37 of 134 |
| Final train / validation loss | ~0.49 / ~0.64 | ~0.22 / ~0.57 |

Recall E2 → E4: Cardboard 0.71 → 0.80, Glass 0.70 → 0.78, Textile 0.51 → 0.59, Plastic 0.75 → 0.81, Paper 0.84 → 0.89, Vegetation 0.94 → 0.99, Miscellaneous 0.70 → 0.71, Food Organics 0.85 → 0.78, Metal 0.93 → 0.87.

Reading (D028):
- Without augmentation the network fits the training set much more closely (larger train–validation loss gap): augmentation does act as a regulariser.
- That regularisation did not raise validation macro-F1 within the 40-epoch budget. E4 scored 0.021 higher, but this is one run per setting and smaller than the ±0.04 epoch-to-epoch validation noise, so the report must not claim that augmentation hurts or helps accuracy here.
- Untested hypotheses for the discussion section: the images are already standardised (one top-down setup, centred objects), so geometric augmentation adds little that validation/test need; colour jitter may blur colour cues that separate materials (e.g. brown cardboard, green vegetation); augmented training usually needs more epochs than this budget.
- E2, not E4, stays the core complex CNN: E1–E3 share one augmentation profile (CONSTITUTION C2), and switching after seeing this result would be post-hoc re-selection.

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
