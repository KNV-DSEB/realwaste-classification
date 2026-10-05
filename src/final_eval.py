"""Final test-set evaluation (CONSTITUTION C3, D018, D026, D029).

For a tuned model the seed runs of its selected configuration (`selection.json`) are evaluated and
summarised as mean ± SD; an untuned model (E4) is evaluated from its single run. Every checkpoint is
evaluated on the test split once: re-running returns the stored result for the same checkpoint, a
changed checkpoint raises (re-evaluating a retrained model needs a DECISION_LOG entry), and a checkpoint
trained on another split is refused.
"""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import get_test_loader, load_class_mapping
from src.evaluate import classification_metrics, per_class_metrics, plot_confusion_matrix, predict
from src.models import build_model
from src.tuning import BASE, summary_path, trial_dirs
from src.utils import environment_info, get_paths, sha256_file

ERROR_COLUMNS = ["experiment_id", "filepath", "true_class", "predicted_class", "confidence",
                 "failure_mode", "visual_note", "reviewer"]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def experiment_dirs(cfg, experiment_id):
    paths = get_paths(cfg)
    return paths.checkpoint_dir / experiment_id, paths.result_dir / "experiments" / experiment_id


def display_name(cfg, experiment_id, run_info):
    return cfg["experiments"].get(experiment_id, {}).get("display_name", run_info["model_name"])


def runs_to_evaluate(cfg, experiment_id):
    """(seed, trial_id) pairs: the seed runs of the selected tuned configuration, or the single base run."""
    _, result_dir = experiment_dirs(cfg, experiment_id)
    selection_path = result_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        return [(int(seed), tid) for seed, tid in selection["seed_trials"].items()]
    if experiment_id in cfg.get("tuning", {}).get("phase1", {}):
        raise RuntimeError(f"{experiment_id} is in the tuning plan (D029) but has no selection.json; finish its tuning first.")
    return [(cfg["project"]["seed"], BASE)]


def evaluate_checkpoint(cfg, experiment_id, tid, out_dir, split_df, device, amp=False):
    """Evaluate one trial's `{id}_best.pt` on the test split (once) and write its files into `out_dir`."""
    checkpoint_dir, _ = trial_dirs(cfg, experiment_id, tid)
    best_path = checkpoint_dir / f"{experiment_id}_best.pt"
    metrics_path = out_dir / f"{experiment_id}_metrics.json"
    checkpoint_sha256 = sha256_file(best_path)

    if metrics_path.exists():
        stored = json.loads(metrics_path.read_text())
        if stored["checkpoint_sha256"] != checkpoint_sha256:
            raise RuntimeError(
                f"{experiment_id} {tid}: {best_path.name} changed after its test evaluation on {stored['evaluated_at']}. "
                "Re-evaluating a changed model needs a DECISION_LOG entry; then move the old metrics file aside."
            )
        print(f"{experiment_id} {tid}: already evaluated on test ({stored['evaluated_at']}); using the stored result.")
        return stored

    checkpoint = torch.load(best_path, map_location=device)
    run_info = checkpoint["run_info"]
    if run_info["split_sha256"] != cfg["split"]["split_sha256"]:
        raise ValueError(f"{experiment_id} {tid} was trained on split {run_info['split_sha256'][:12]}…, "
                         f"not the frozen split {cfg['split']['split_sha256'][:12]}…")
    class_names = load_class_mapping(cfg)["class_name"].tolist()
    if run_info["class_names"] != class_names:
        raise ValueError(f"{experiment_id} {tid}: class order in the checkpoint differs from class_mapping.csv")
    summary = json.loads(summary_path(cfg, experiment_id, tid).read_text())

    model = build_model(run_info["model_name"], **run_info["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model"])
    preds = predict(model, get_test_loader(cfg, split_df), device, amp=amp)
    metrics = classification_metrics(preds.y_true, preds.y_pred, len(class_names))
    name = display_name(cfg, experiment_id, run_info)
    seed = summary.get("seed", cfg["project"]["seed"])

    predictions = pd.DataFrame({
        "filepath": preds.filepaths,
        "true_class": [class_names[i] for i in preds.y_true],
        "predicted_class": [class_names[i] for i in preds.y_pred],
        "confidence": preds.probs.max(axis=1).round(4),
    })
    predictions.to_csv(out_dir / f"{experiment_id}_predictions.csv", index=False)
    errors = predictions[predictions["true_class"] != predictions["predicted_class"]]
    errors.assign(experiment_id=experiment_id, failure_mode="", visual_note="", reviewer="")[ERROR_COLUMNS] \
        .sort_values(["true_class", "predicted_class", "confidence"], ascending=[True, True, False]) \
        .to_csv(out_dir / f"{experiment_id}_errors.csv", index=False)
    per_class_metrics(preds.y_true, preds.y_pred, class_names) \
        .to_csv(out_dir / f"{experiment_id}_per_class_metrics.csv", index=False)
    fig = plot_confusion_matrix(preds.y_true, preds.y_pred, class_names,
                                f"{experiment_id} {name} (seed {seed}) - test confusion matrix")
    fig.savefig(out_dir / f"{experiment_id}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "experiment_id": experiment_id,
        "model_name": name,
        "architecture": run_info["model_name"],
        "trial_id": tid,
        "seed": seed,
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
    print(f"{experiment_id} {name} {tid} (seed {seed}): test accuracy {metrics['accuracy']:.4f}, macro-F1 {metrics['macro_f1']:.4f}")
    return result


def evaluate_experiment(cfg, experiment_id, split_df, device, amp=False):
    """Evaluate every run of an experiment. The base-seed run is the primary one: its files go to
    `04_Results/experiments/{id}/`, other seeds to `.../test_seeds/seed{n}/`."""
    _, result_dir = experiment_dirs(cfg, experiment_id)
    runs = []
    for seed, tid in runs_to_evaluate(cfg, experiment_id):
        out_dir = result_dir if seed == cfg["project"]["seed"] else result_dir / "test_seeds" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        runs.append(evaluate_checkpoint(cfg, experiment_id, tid, out_dir, split_df, device, amp))
    frame = pd.DataFrame(runs)
    keys = METRICS + ["best_val_macro_f1"]
    aggregate = {
        "experiment_id": experiment_id,
        "model_name": runs[0]["model_name"],
        "trial_id": runs[0]["trial_id"],
        "seeds": [int(r["seed"]) for r in runs],
        "mean": {k: float(frame[k].mean()) for k in keys},
        "sd": {k: (float(frame[k].std(ddof=1)) if len(runs) > 1 else None) for k in keys},
        "parameter_count": runs[0]["parameter_count"],
        "primary": runs[0],
    }
    if len(runs) > 1:
        (result_dir / f"{experiment_id}_metrics_seeds.json").write_text(
            json.dumps({**{k: v for k, v in aggregate.items() if k != "primary"},
                        "per_seed": [{k: r[k] for k in ["seed", "trial_id"] + keys} for r in runs]}, indent=2))
    return aggregate


def comparison_table(aggregates):
    """Main report table (EXPERIMENT_PROTOCOL): test metrics as mean and SD over the evaluated seeds."""
    rows = []
    for a in aggregates:
        row = {"Experiment": a["experiment_id"], "Model": a["model_name"], "Trial": a["trial_id"], "Seeds": len(a["seeds"])}
        for key, label in zip(METRICS, ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]):
            row[label] = round(a["mean"][key], 4)
            row[f"{label} SD"] = None if a["sd"][key] is None else round(a["sd"][key], 4)
        row["Val Macro F1"] = round(a["mean"]["best_val_macro_f1"], 4)
        row["Parameters"] = a["parameter_count"]
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_f1_table(cfg, experiment_ids):
    """Test F1 per class (rows) and experiment (columns), from each experiment's primary run."""
    columns = {}
    for experiment_id in experiment_ids:
        _, result_dir = experiment_dirs(cfg, experiment_id)
        per_class = pd.read_csv(result_dir / f"{experiment_id}_per_class_metrics.csv")
        columns[experiment_id] = per_class.set_index("class_name")["f1"]
    return pd.DataFrame(columns)
