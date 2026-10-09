"""Final evaluation on the test set (notebook 09). Each checkpoint is scored once; the result is stored
and reused, and the run stops if a checkpoint was changed after it was evaluated."""
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.dataset import get_test_loader, load_class_mapping
from src.evaluate import classification_metrics, per_class_metrics, plot_confusion_matrix, predict
from src.models import build_model
from src.tuning import trial_dirs
from src.utils import get_paths, sha256_file

METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def evaluate_run(cfg, exp, tid, out_dir, split_df, device, amp=False):
    ckpt_dir, res_dir = trial_dirs(cfg, exp, tid)
    ckpt_path = ckpt_dir / f"{exp}_best.pt"
    out_file = out_dir / f"{exp}_metrics.json"
    sha = sha256_file(ckpt_path)
    if out_file.exists():
        stored = json.loads(out_file.read_text())
        if stored["checkpoint_sha256"] != sha:
            raise RuntimeError(f"{ckpt_path} changed after it was evaluated on the test set")
        return stored

    ckpt = torch.load(ckpt_path, map_location=device)
    info = ckpt["run_info"]
    if info["split_sha256"] != cfg["split"]["split_sha256"]:
        raise ValueError(f"{ckpt_path} was trained on a different split")
    classes = load_class_mapping(cfg)["class_name"].tolist()
    model = build_model(info["model_name"], **info["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model"])
    p = predict(model, get_test_loader(cfg, split_df), device, amp=amp)
    name = cfg["experiments"][exp].get("display_name", info["model_name"])

    preds = pd.DataFrame({"filepath": p.filepaths, "true_class": [classes[i] for i in p.y_true],
                          "predicted_class": [classes[i] for i in p.y_pred], "confidence": p.probs.max(1).round(4)})
    preds.to_csv(out_dir / f"{exp}_predictions.csv", index=False)
    errors = preds[preds["true_class"] != preds["predicted_class"]].sort_values("confidence", ascending=False)
    errors.assign(failure_mode="", visual_note="").to_csv(out_dir / f"{exp}_errors.csv", index=False)
    per_class_metrics(p.y_true, p.y_pred, classes).to_csv(out_dir / f"{exp}_per_class_metrics.csv", index=False)
    fig = plot_confusion_matrix(p.y_true, p.y_pred, classes, f"{exp} {name} - test")
    fig.savefig(out_dir / f"{exp}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = json.loads((res_dir / f"{exp}_training_summary.json").read_text())
    result = {"experiment_id": exp, "model_name": name, "trial_id": tid,
              "seed": summary.get("seed", cfg["project"]["seed"]), "eval_split": "test",
              **classification_metrics(p.y_true, p.y_pred, len(classes)),
              "best_val_macro_f1": summary["best_val_macro_f1"], "best_epoch": summary["best_epoch"],
              "parameter_count": summary["parameter_count"], "checkpoint_sha256": sha,
              "evaluated_at": datetime.now().isoformat(timespec="seconds")}
    out_file.write_text(json.dumps(result, indent=2))
    print(f"{exp} {tid}: test accuracy {result['accuracy']:.4f}, macro-F1 {result['macro_f1']:.4f}")
    return result


def evaluate_experiment(cfg, exp, split_df, device, amp=False):
    """All seed runs listed in selection.json (notebook 10); without it only the seed-42 run.
    Stored results are reused, so running this again after notebook 10 only adds the new seeds."""
    res_dir = trial_dirs(cfg, exp, "base")[1]
    selection = res_dir / "selection.json"
    if selection.exists():
        runs = {int(s): t for s, t in json.loads(selection.read_text())["seed_trials"].items()}
    else:
        if exp in cfg["tuning"]["phase1"]:
            print(f"{exp}: seeds 43/44 not run yet (notebook 10), evaluating seed 42 only")
        runs = {cfg["project"]["seed"]: "base"}

    results = []
    for seed, tid in runs.items():
        # seed 42 is the main run; other seeds go to their own folder
        out_dir = res_dir if seed == cfg["project"]["seed"] else res_dir / "test_seeds" / f"seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results.append(evaluate_run(cfg, exp, tid, out_dir, split_df, device, amp))
    return results


def comparison_table(results):
    """Mean and SD over seeds for each experiment; results = {exp: [run results]}."""
    rows = []
    for exp, runs in results.items():
        df = pd.DataFrame(runs)
        row = {"experiment": exp, "model": runs[0]["model_name"], "seeds": len(runs)}
        for m in METRICS + ["best_val_macro_f1"]:
            row[m] = round(df[m].mean(), 4)
            row[m + "_sd"] = round(df[m].std(), 4) if len(runs) > 1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def per_class_f1_table(cfg, experiments):
    res = get_paths(cfg).result_dir / "experiments"
    return pd.DataFrame({e: pd.read_csv(res / e / f"{e}_per_class_metrics.csv").set_index("class_name")["f1"]
                         for e in experiments})
