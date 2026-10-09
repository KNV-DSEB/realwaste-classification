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

Written by notebook 10 (validation only; D029, D030 — seed runs of the existing settings):
- one folder per trial: `03_Checkpoints/{EXPERIMENT_ID}/trials/{trial}/` and `04_Results/experiments/{EXPERIMENT_ID}/trials/{trial}/`, with the same file names as a normal run; the original run is trial `base` and stays in place
- `04_Results/experiments/{EXPERIMENT_ID}/selection.json` (selected trial, its seed runs, validation macro-F1 per seed, mean ± SD), `{EXPERIMENT_ID}_seed_runs.csv` (every run)

Written once by the final-evaluation notebook (test set, D018, D026, D029). For E1–E3 the files below are for the seed-42 run; seeds 43/44 go to `test_seeds/seed{n}/`:
- `{EXPERIMENT_ID}_metrics.json`
- `{EXPERIMENT_ID}_predictions.csv`
- `{EXPERIMENT_ID}_per_class_metrics.csv`
- `{EXPERIMENT_ID}_confusion_matrix.png`
- `{EXPERIMENT_ID}_errors.csv` — misclassified test images, most confident first: `filepath, true_class, predicted_class, confidence, failure_mode, visual_note` (the last two are filled in by hand)
- `04_Results/final/model_comparison.csv` (mean and SD over seeds per experiment), `per_class_f1.csv` (seed-42 runs)

## metrics.json fields
Final test-set metrics of the validation-selected checkpoint.
- experiment_id, model_name, trial_id, seed
- eval_split — always `test`
- accuracy, macro_precision, macro_recall, macro_f1
- best_val_macro_f1, best_epoch (selection by validation macro-F1, D010)
- parameter_count
- checkpoint_sha256 — a re-run stops if the checkpoint changed after it was evaluated
- evaluated_at

The run's config, split hash and environment are in its `{EXPERIMENT_ID}_training_summary.json` and in `run_info` inside the checkpoint.

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
