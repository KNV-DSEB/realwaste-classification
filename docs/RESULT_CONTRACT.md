# RESULT_CONTRACT

Every core experiment must create consistent outputs.

## Minimum files
- `{EXPERIMENT_ID}_metrics.json`
- `{EXPERIMENT_ID}_history.csv`
- `{EXPERIMENT_ID}_predictions.csv`
- `{EXPERIMENT_ID}_confusion_matrix.png`
- `{EXPERIMENT_ID}_best.pt`

## metrics.json fields
- experiment_id
- model_name
- seed
- best_epoch
- accuracy
- macro_precision
- macro_recall
- macro_f1
- parameter_count
- training_time_minutes
- checkpoint_path
- config_snapshot
- notes

## predictions.csv fields
- filepath
- true_class
- predicted_class
- confidence

## Error-analysis rule
Preserve predictions for the final test set so misclassification examples can be grouped by confusion type and inspected visually.
