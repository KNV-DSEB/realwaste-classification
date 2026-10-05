# RESULT_CONTRACT

Every core experiment must create consistent outputs.

## Naming (D012)
- `EXPERIMENT_ID` ∈ {`E1`, `E2`, `E3`, `E4`}; `model_name` ∈ {`SimpleCNN`, `MultiScaleCNN`, `EfficientNet-B0`, `MultiScaleCNN_NoAug`}.
- Checkpoint: `03_Checkpoints/{EXPERIMENT_ID}/{EXPERIMENT_ID}_best.pt`
- Result files: `04_Results/experiments/{EXPERIMENT_ID}/`

## Minimum files
Written by the training notebook (validation only):
- `{EXPERIMENT_ID}_best.pt`, `{EXPERIMENT_ID}_last.pt` (checkpoints)
- `{EXPERIMENT_ID}_history.csv`
- `{EXPERIMENT_ID}_curves.png` — training/validation loss and accuracy
- `{EXPERIMENT_ID}_val_per_class_metrics.csv`, `{EXPERIMENT_ID}_val_confusion_matrix.png`
- `{EXPERIMENT_ID}_training_summary.json` — run record (EXPERIMENT_PROTOCOL "Training records")
- E3 only: per-stage files `E3_stageA_best.pt`, `E3_stageA_last.pt`, `E3_stageB_best.pt`, `E3_stageB_last.pt`, `E3_stageA_history.csv`, `E3_stageB_history.csv`; `E3_best.pt` is a copy of the better stage-best checkpoint (D010, D022)

Written by the tuning notebook (validation only, D029):
- one folder per trial: `03_Checkpoints/{EXPERIMENT_ID}/trials/{trial}/` and `04_Results/experiments/{EXPERIMENT_ID}/trials/{trial}/`, with the same file names as a normal run; the original run is trial `base` and stays in place
- `04_Results/experiments/{EXPERIMENT_ID}/selection.json` (selected trial, its seed runs, validation macro-F1 per seed, mean ± SD), `{EXPERIMENT_ID}_tuning.csv` (every trial), `{EXPERIMENT_ID}_tuning_curves.png`

Written once by the final-evaluation notebook (test set, D018, D026, D029). For a tuned model the files below are for the seed-42 run of the selected configuration; seeds 43/44 go to `test_seeds/seed{n}/` and `{EXPERIMENT_ID}_metrics_seeds.json` holds mean ± SD:
- `{EXPERIMENT_ID}_metrics.json`
- `{EXPERIMENT_ID}_predictions.csv`
- `{EXPERIMENT_ID}_per_class_metrics.csv`
- `{EXPERIMENT_ID}_confusion_matrix.png`
- `{EXPERIMENT_ID}_errors.csv` — misclassified test images with the `templates/error_analysis_template.csv` columns
- `04_Results/final/model_comparison.csv`, `per_class_f1_comparison.csv`, `{primary}_test_error_examples.png`

## metrics.json fields
Final test-set metrics of the validation-selected checkpoint.
- experiment_id
- model_name
- seed
- eval_split — always `test`
- selection_metric — `val_macro_f1` (D010)
- best_val_macro_f1
- best_epoch
- accuracy
- macro_precision
- macro_recall
- macro_f1
- parameter_count
- training_time_minutes
- checkpoint_path
- split_sha256
- class_mapping_sha256
- config_snapshot — the full config plus `src.utils.environment_info()` (library versions, GPU, git commit)
- notes
- also written: `architecture`, `trial_id`, `test_images`, `checkpoint_sha256` (guards against re-evaluating a changed checkpoint), `evaluated_at`

## history.csv fields
One row per epoch: `epoch, train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, lr, seconds`. E3 adds `stage` (`A` or `B`). Training loss/accuracy are running averages over augmented images with dropout active.

## per_class_metrics.csv fields
Test set: `class_index, class_name, support, precision, recall, f1`.

## predictions.csv fields
Test set, one row per image:
- filepath — the `split.csv` row ID
- true_class
- predicted_class
- confidence

Class names are exactly those in `class_mapping.csv`.

## Error-analysis rule
Preserve predictions for the final test set so misclassification examples can be grouped by confusion type and inspected visually.
