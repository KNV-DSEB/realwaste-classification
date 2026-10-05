"""Validation-only hyperparameter tuning of the core models (D029).

Each core model gets the same number of trials per phase (`configs/config.yaml` → `tuning`):
  phase 1 — learning rate;
  phase 2 — regularisation (E1/E2) or fine-tuning depth (E3), on top of the phase-1 winner;
  phase 3 — the phase-1/2 winner re-run with extra seeds.
The existing run of each model is the trial "base" (seed 42) and stays where it is. Other trials write to
`03_Checkpoints/{id}/trials/{trial}/` and `04_Results/experiments/{id}/trials/{trial}/` with the same file
names as a normal run, so finished trials are skipped and interrupted ones resume. Winners are chosen by
validation macro F1 only; the test split is never loaded here.
"""
import copy
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from src.dataset import get_dataloaders, load_class_mapping
from src.models import build_model
from src.train import FitResult, build_optimizer, build_scheduler, fit, plot_history
from src.utils import count_parameters, environment_info, get_paths, set_seed

BASE = "base"
MODEL_KWARGS = {
    "SimpleCNN": ["dropout"],
    "MultiScaleCNN": ["dropout", "stem_width", "widths"],
    "EfficientNet-B0": ["dropout"],
}
SCRATCH_EPOCHS_KEY = {"E1": "epochs_simple", "E2": "epochs_complex"}
SHORT_NAMES = {"lr": "lr", "weight_decay": "wd", "stage_b.lr": "lrB", "stage_b.trainable_blocks": "blocksB"}


def deep_merge(base, overrides):
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _flatten(d, prefix=""):
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def trial_id(overrides, seed, base_seed):
    """Readable, unique folder name, e.g. "lr3e-04_wd1e-02", "lrB3e-04_blocksB9_seed43", or "base"."""
    parts = []
    for key, value in sorted(_flatten(overrides).items()):
        label = SHORT_NAMES.get(key, key.replace(".", "-"))
        parts.append(f"{label}{value:.0e}" if isinstance(value, float) else f"{label}{value}")
    if seed != base_seed:
        parts.append(f"seed{seed}")
    return "_".join(parts) or BASE


def trial_dirs(cfg, experiment_id, tid):
    paths = get_paths(cfg)
    checkpoint_dir = paths.checkpoint_dir / experiment_id
    result_dir = paths.result_dir / "experiments" / experiment_id
    if tid == BASE:
        return checkpoint_dir, result_dir
    return checkpoint_dir / "trials" / tid, result_dir / "trials" / tid


def summary_path(cfg, experiment_id, tid):
    return trial_dirs(cfg, experiment_id, tid)[1] / f"{experiment_id}_training_summary.json"


# ----------------------------------------------------------------------------- training one trial

def _scratch(cfg, experiment_id, exp_cfg, run_info, seed, loaders, checkpoint_dir, result_dir, device, amp):
    train_loader, val_loader = loaders
    set_seed(seed)
    model = build_model(exp_cfg["model_name"], **run_info["model_kwargs"]).to(device)
    optimizer = build_optimizer(model.parameters(), exp_cfg)
    result = fit(
        model, train_loader, val_loader,
        optimizer=optimizer, scheduler=build_scheduler(optimizer, exp_cfg), criterion=nn.CrossEntropyLoss(),
        device=device, amp=amp, num_classes=run_info["model_kwargs"]["num_classes"],
        max_epochs=cfg["training"][SCRATCH_EPOCHS_KEY[experiment_id]],
        patience=cfg["training"]["early_stopping_patience"],
        checkpoint_dir=checkpoint_dir, experiment_id=experiment_id,
        history_path=result_dir / f"{experiment_id}_history.csv", run_info=run_info,
    )
    best = torch.load(result.best_path, map_location="cpu")
    return result.history, {
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_val_macro_f1,
        "best_val_metrics": best["val_metrics"],
        "epochs_completed": int(result.history["epoch"].max()),
        "stopped_early": result.stopped_early,
        "training_time_minutes": round(float(result.history["seconds"].sum()) / 60, 1),
        "parameter_count": count_parameters(model),
        "checkpoint_path": str(result.best_path),
    }


def _transfer(cfg, experiment_id, exp_cfg, run_info, seed, overrides, loaders, checkpoint_dir, result_dir, device, amp):
    """Two-stage E3 trial. A trial that changes only stage B with the base seed starts from the base run's
    best stage-A checkpoint instead of retraining stage A (identical settings)."""
    train_loader, val_loader = loaders
    num_classes = run_info["model_kwargs"]["num_classes"]
    criterion = nn.CrossEntropyLoss()
    set_seed(seed)
    model = build_model(exp_cfg["model_name"], pretrained=True, **run_info["model_kwargs"]).to(device)

    def run_stage(stage):
        stage_cfg = exp_cfg[f"stage_{stage.lower()}"]
        model.set_trainable_blocks(stage_cfg["trainable_blocks"])
        stage_exp_cfg = {**exp_cfg, "lr": stage_cfg["lr"]}
        optimizer = build_optimizer([p for p in model.parameters() if p.requires_grad], stage_exp_cfg)
        return fit(
            model, train_loader, val_loader,
            optimizer=optimizer, scheduler=build_scheduler(optimizer, stage_exp_cfg), criterion=criterion,
            device=device, amp=amp, num_classes=num_classes,
            max_epochs=cfg["training"]["epochs_transfer_head" if stage == "A" else "epochs_finetune"],
            patience=cfg["training"]["early_stopping_patience"],
            checkpoint_dir=checkpoint_dir, experiment_id=f"{experiment_id}_stage{stage}",
            history_path=result_dir / f"{experiment_id}_stage{stage}_history.csv",
            run_info={**run_info, "stage": stage},
        )

    reuse_stage_a = set(overrides) <= {"stage_b"} and seed == cfg["tuning"]["base_seed"]
    if reuse_stage_a:
        base_checkpoint_dir, base_result_dir = trial_dirs(cfg, experiment_id, BASE)
        base_summary = json.loads(summary_path(cfg, experiment_id, BASE).read_text())
        a_best = base_checkpoint_dir / f"{experiment_id}_stageA_best.pt"
        a_state = torch.load(a_best, map_location="cpu")
        result_a = FitResult(
            pd.read_csv(base_result_dir / f"{experiment_id}_stageA_history.csv"),
            a_state["epoch"], round(a_state["val_metrics"]["macro_f1"], 5),
            base_summary["stages"]["A"]["stopped_early"], a_best,
            base_checkpoint_dir / f"{experiment_id}_stageA_last.pt", a_state["run_info"],
        )
        print(f"Stage A reused from the base run (best epoch {result_a.best_epoch}, "
              f"validation macro-F1 {result_a.best_val_macro_f1:.4f})")
    else:
        result_a = run_stage("A")
    model.load_state_dict(torch.load(result_a.best_path, map_location=device)["model"])
    result_b = run_stage("B")

    selected_stage, selected = ("B", result_b) if result_b.best_val_macro_f1 > result_a.best_val_macro_f1 else ("A", result_a)
    final = torch.load(selected.best_path, map_location="cpu")
    final["run_info"] = {**run_info, "selected_stage": selected_stage}
    final_path = checkpoint_dir / f"{experiment_id}_best.pt"
    torch.save(final, final_path)

    stage_a_epochs = len(result_a.history)
    history = pd.concat([
        result_a.history.assign(stage="A", stage_epoch=result_a.history["epoch"]),
        result_b.history.assign(stage="B", stage_epoch=result_b.history["epoch"],
                                epoch=result_b.history["epoch"] + stage_a_epochs),
    ], ignore_index=True)
    history.to_csv(result_dir / f"{experiment_id}_history.csv", index=False)

    def stage_record(stage, result):
        stage_cfg = exp_cfg[f"stage_{stage.lower()}"]
        return {
            "trainable_blocks": stage_cfg["trainable_blocks"], "learning_rate": stage_cfg["lr"],
            "epochs_completed": int(result.history["epoch"].max()), "stopped_early": result.stopped_early,
            "best_epoch": result.best_epoch, "best_val_macro_f1": result.best_val_macro_f1,
            "training_time_minutes": round(float(result.history["seconds"].sum()) / 60, 1),
            "checkpoint_path": str(result.best_path),
            "reused_from_base": stage == "A" and reuse_stage_a,
        }

    best_epoch = int(history.loc[(history["stage"] == selected_stage) & (history["stage_epoch"] == selected.best_epoch),
                                 "epoch"].iloc[0])
    return history, {
        "stages": {"A": stage_record("A", result_a), "B": stage_record("B", result_b)},
        "selected_stage": selected_stage,
        "best_epoch": best_epoch,
        "best_val_macro_f1": selected.best_val_macro_f1,
        "best_val_metrics": final["val_metrics"],
        "epochs_completed": int(history["epoch"].max()),
        "stopped_early": result_b.stopped_early,
        "training_time_minutes": round(float(history["seconds"].sum()) / 60, 1),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_path": str(final_path),
    }


def run_trial(cfg, experiment_id, overrides, seed, phase, split_df, device, amp):
    """Train one trial (or skip it when its summary exists) and return its summary."""
    base_seed = cfg["tuning"]["base_seed"]
    tid = trial_id(overrides, seed, base_seed)
    path = summary_path(cfg, experiment_id, tid)
    if path.exists():
        print(f"{experiment_id} {tid}: already finished, skipped")
        return json.loads(path.read_text())

    print(f"===== {experiment_id} trial {tid} (phase {phase}, seed {seed}, overrides {overrides})")
    exp_cfg = deep_merge(cfg["experiments"][experiment_id], overrides)
    run_cfg = deep_merge(cfg, {"project": {"seed": seed}})  # seeds the shuffling generator too
    checkpoint_dir, result_dir = trial_dirs(cfg, experiment_id, tid)
    result_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_mapping(cfg)["class_name"].tolist()
    loaders = get_dataloaders(run_cfg, augment=exp_cfg.get("augment", True), split_df=split_df)
    env = environment_info()
    run_info = {
        "experiment_id": experiment_id,
        "run_id": f"{experiment_id}_{tid}_{datetime.now():%Y%m%d-%H%M%S}",
        "trial_id": tid, "phase": phase, "overrides": overrides, "seed": seed,
        "model_name": exp_cfg["model_name"],
        "model_kwargs": {"num_classes": len(class_names), **{k: exp_cfg[k] for k in MODEL_KWARGS[exp_cfg["model_name"]]}},
        "class_names": class_names,
        "split_sha256": cfg["split"]["split_sha256"],
        "git_commit": env["git_commit"],
    }
    if experiment_id == "E3":
        history, record = _transfer(cfg, experiment_id, exp_cfg, run_info, seed, overrides, loaders,
                                    checkpoint_dir, result_dir, device, amp)
    else:
        history, record = _scratch(cfg, experiment_id, exp_cfg, run_info, seed, loaders,
                                   checkpoint_dir, result_dir, device, amp)

    fig = plot_history(history, f"{experiment_id} {exp_cfg['model_name']} - trial {tid}")
    fig.savefig(result_dir / f"{experiment_id}_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    summary = {
        **run_info, **record,
        "selection_metric": cfg["training"]["selection_metric"],
        "augmentation": "D011 training profile" if exp_cfg.get("augment", True) else "none",
        "class_weighting": cfg["data"]["class_weighting"],
        "amp": amp,
        "experiment_config": exp_cfg,
        "environment": env,
        "notes": "",
    }
    path.write_text(json.dumps(summary, indent=2))  # written last: its presence marks the trial as finished
    print(f"{experiment_id} {tid}: best validation macro-F1 {summary['best_val_macro_f1']:.4f} at epoch {summary['best_epoch']}")
    return summary


# ----------------------------------------------------------------------------- tables, phases, selection

def tuning_table(cfg, experiment_id):
    """One row per finished trial of the experiment, base run included."""
    base_seed = cfg["tuning"]["base_seed"]
    _, result_dir = trial_dirs(cfg, experiment_id, BASE)
    paths = [summary_path(cfg, experiment_id, BASE)]
    paths += sorted((result_dir / "trials").glob(f"*/{experiment_id}_training_summary.json"))
    rows = []
    for path in paths:
        s = json.loads(path.read_text())
        stages = s.get("stages")
        rows.append({
            "trial_id": s.get("trial_id", BASE),
            "phase": s.get("phase", 0),
            "seed": s.get("seed", base_seed),
            "overrides": json.dumps(s.get("overrides", {}), sort_keys=True),
            "val_macro_f1": s["best_val_macro_f1"],
            "val_accuracy": round(s["best_val_metrics"]["accuracy"], 5),
            "best_epoch": s["best_epoch"],
            "epochs_completed": s.get("epochs_completed") or sum(st["epochs_completed"] for st in stages.values()),
            "stopped_early": s["stopped_early"] if "stopped_early" in s else stages["B"]["stopped_early"],
            "training_time_minutes": s["training_time_minutes"],
        })
    return pd.DataFrame(rows).sort_values(["phase", "trial_id"], kind="stable").reset_index(drop=True)


def winner(cfg, table, max_phase):
    """Base-seed trial with the best validation macro F1 up to `max_phase` (ties keep the earlier phase)."""
    candidates = table[(table["seed"] == cfg["tuning"]["base_seed"]) & (table["phase"] <= max_phase)]
    return candidates.sort_values(["val_macro_f1", "phase"], ascending=[False, True], kind="stable").iloc[0]


def plan_phase(cfg, experiment_id, phase, table):
    """(overrides, seed) of every trial in `phase`; raises if the previous phase is unfinished."""
    tuning = cfg["tuning"]
    base_seed = tuning["base_seed"]
    if phase == 1:
        return [(o, base_seed) for o in tuning["phase1"][experiment_id]]
    _require_finished(cfg, experiment_id, phase - 1, table)
    if phase == 2:
        start = json.loads(winner(cfg, table, 1)["overrides"])
        return [(deep_merge(start, o), base_seed) for o in tuning["phase2"][experiment_id]]
    if phase == 3:
        best = json.loads(winner(cfg, table, 2)["overrides"])
        return [(best, s) for s in tuning["phase3_seeds"]]
    raise ValueError(f"Unknown phase {phase}")


def _require_finished(cfg, experiment_id, phase, table):
    planned = {trial_id(o, s, cfg["tuning"]["base_seed"]) for o, s in plan_phase(cfg, experiment_id, phase, table)}
    missing = planned - set(table["trial_id"])
    if missing:
        raise RuntimeError(f"{experiment_id}: phase {phase} is not finished (missing trials: {sorted(missing)})")


def write_selection(cfg, experiment_id):
    """After phase 3: record the selected configuration and its seed runs for the final evaluation."""
    tuning = cfg["tuning"]
    table = tuning_table(cfg, experiment_id)
    _require_finished(cfg, experiment_id, 3, table)
    best = winner(cfg, table, 2)
    overrides = json.loads(best["overrides"])
    seeds = [tuning["base_seed"]] + tuning["phase3_seeds"]
    seed_trials = {str(s): trial_id(overrides, s, tuning["base_seed"]) for s in seeds}
    val = {s: float(table.loc[table["trial_id"] == t, "val_macro_f1"].iloc[0]) for s, t in seed_trials.items()}
    values = pd.Series(list(val.values()))
    selection = {
        "experiment_id": experiment_id,
        "selected_trial": best["trial_id"],
        "overrides": overrides,
        "seed_trials": seed_trials,
        "val_macro_f1_by_seed": val,
        "val_macro_f1_mean": round(float(values.mean()), 5),
        "val_macro_f1_std": round(float(values.std(ddof=1)), 5),
        "selected_at": datetime.now().isoformat(timespec="seconds"),
    }
    _, result_dir = trial_dirs(cfg, experiment_id, BASE)
    (result_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    table.to_csv(result_dir / f"{experiment_id}_tuning.csv", index=False)
    return selection
