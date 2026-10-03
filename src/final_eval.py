"""Final test-set evaluation (CONSTITUTION C3, D018, D026).

Each frozen, validation-selected checkpoint `{id}_best.pt` is evaluated on the test split once and
its RESULT_CONTRACT files are written. Re-running returns the stored result for the same checkpoint;
a changed checkpoint raises, because re-evaluating a retrained model needs a DECISION_LOG entry.
"""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import get_test_loader, load_class_mapping
from src.evaluate import classification_metrics, per_class_metrics, plot_confusion_matrix, predict
from src.models import build_model
from src.utils import environment_info, get_paths, sha256_file

ERROR_COLUMNS = ["experiment_id", "filepath", "true_class", "predicted_class", "confidence",
                 "failure_mode", "visual_note", "reviewer"]


def experiment_dirs(cfg, experiment_id):
    paths = get_paths(cfg)
    return paths.checkpoint_dir / experiment_id, paths.result_dir / "experiments" / experiment_id


def display_name(cfg, experiment_id, run_info):
    return cfg["experiments"].get(experiment_id, {}).get("display_name", run_info["model_name"])


def evaluate_on_test(cfg, experiment_id, split_df, device, amp=False):
    """Evaluate `{id}_best.pt` on the test split (once) and write metrics, predictions, per-class
    metrics, confusion matrix and the list of misclassified images for error analysis."""
    checkpoint_dir, result_dir = experiment_dirs(cfg, experiment_id)
    best_path = checkpoint_dir / f"{experiment_id}_best.pt"
    metrics_path = result_dir / f"{experiment_id}_metrics.json"
    checkpoint_sha256 = sha256_file(best_path)

    if metrics_path.exists():
        stored = json.loads(metrics_path.read_text())
        if stored["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError(
                f"{experiment_id}: {best_path.name} changed after its test evaluation on {stored['evaluated_at']}. "
                "Re-evaluating a changed model needs a DECISION_LOG entry; then move the old metrics file aside."
            )
        print(f"{experiment_id}: already evaluated on test ({stored['evaluated_at']}); using the stored result.")
        return stored

    checkpoint = torch.load(best_path, map_location=device)
    run_info = checkpoint["run_info"]
    if run_info["split_sha256"] != cfg["split"]["split_sha256"]:
        raise ValueError(f"{experiment_id} was trained on split {run_info['split_sha256'][:12]}…, "
                         f"not the frozen split {cfg['split']['split_sha256'][:12]}…")
    class_names = load_class_mapping(cfg)["class_name"].tolist()
    if run_info["class_names"] != class_names:
        raise ValueError(f"{experiment_id}: class order in the checkpoint differs from class_mapping.csv")
    summary = json.loads((result_dir / f"{experiment_id}_training_summary.json").read_text())

    model = build_model(run_info["model_name"], **run_info["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    preds = predict(model, get_test_loader(cfg, split_df), device, amp=amp)
    metrics = classification_metrics(preds.y_true, preds.y_pred, len(class_names))
    name = display_name(cfg, experiment_id, run_info)

    predictions = pd.DataFrame({
        "filepath": preds.filepaths,
        "true_class": [class_names[i] for i in preds.y_true],
        "predicted_class": [class_names[i] for i in preds.y_pred],
        "confidence": preds.probs.max(axis=1).round(4),
    })
    predictions.to_csv(result_dir / f"{experiment_id}_predictions.csv", index=False)
    errors = predictions[predictions["true_class"] != predictions["predicted_class"]]
    errors.assign(experiment_id=experiment_id, failure_mode="", visual_note="", reviewer="")[ERROR_COLUMNS] \
        .sort_values(["true_class", "predicted_class", "confidence"], ascending=[True, True, False]) \
        .to_csv(result_dir / f"{experiment_id}_errors.csv", index=False)
    per_class_metrics(preds.y_true, preds.y_pred, class_names) \
        .to_csv(result_dir / f"{experiment_id}_per_class_metrics.csv", index=False)
    fig = plot_confusion_matrix(preds.y_true, preds.y_pred, class_names, f"{experiment_id} {name} - test confusion matrix")
    fig.savefig(result_dir / f"{experiment_id}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "experiment_id": experiment_id,
        "model_name": name,
        "architecture": run_info["model_name"],
        "seed": summary["seed"],
        "eval_split": "test",
        "test_images": len(predictions),
        "selection_metric": summary["selection_metric"],
        "best_val_macro_f1": summary["best_val_macro_f1"],
        "best_epoch": summary["best_epoch"],
        **metrics,
        "parameter_count": summary["parameter_count"],
        "training_time_minutes": summary["training_time_minutes"],
        "checkpoint_path": str(best_path),
        "checkpoint_sha256": checkpoint_sha256,
        "split_sha256": cfg["split"]["split_sha256"],
        "class_mapping_sha256": cfg["split"]["class_mapping_sha256"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "config_snapshot": {"config": cfg, "environment": environment_info(), "run_info": run_info},
        "notes": "",
    }
    metrics_path.write_text(json.dumps(result, indent=2))
    print(f"{experiment_id} {name}: test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")
    return result


def comparison_table(results):
    """Main report table (EXPERIMENT_PROTOCOL): one row per evaluated experiment."""
    return pd.DataFrame([{
        "Experiment": r["experiment_id"],
        "Model": r["model_name"],
        "Accuracy": round(r["accuracy"], 4),
        "Macro Precision": round(r["macro_precision"], 4),
        "Macro Recall": round(r["macro_recall"], 4),
        "Macro F1": round(r["macro_f1"], 4),
        "Val Macro F1": round(r["best_val_macro_f1"], 4),
        "Parameters": r["parameter_count"],
    } for r in results])


def per_class_f1_table(cfg, experiment_ids):
    """Test F1 per class (rows) and experiment (columns)."""
    columns = {}
    for experiment_id in experiment_ids:
        _, result_dir = experiment_dirs(cfg, experiment_id)
        per_class = pd.read_csv(result_dir / f"{experiment_id}_per_class_metrics.csv")
        columns[experiment_id] = per_class.set_index("class_name")["f1"]
    return pd.DataFrame(columns)
