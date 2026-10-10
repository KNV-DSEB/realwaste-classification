"""Extra training runs on the validation set (notebook 10).

Written for a 3-phase search (D029: learning rate, then weight decay / fine-tuned blocks, then seeds).
Since D030 phases 1-2 are empty in config.yaml, so only phase 3 runs: the existing settings with seeds 43 and 44.
The original run of each model counts as the trial "base" (seed 42).
"""
import copy
import json
from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from src.dataset import get_dataloaders, load_class_mapping
from src.models import build_model
from src.train import build_optimizer, build_scheduler, fit
from src.utils import count_parameters, get_paths, set_seed

MODEL_ARGS = {"SimpleCNN": ["dropout"], "MultiScaleCNN": ["dropout", "stem_width", "widths"],
              "EfficientNet-B0": ["dropout"]}
SHORT = {"lr": "lr", "weight_decay": "wd", "stage_b.lr": "lrB", "stage_b.trainable_blocks": "blocksB"}


def merge(base, extra):
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {prefix + k: v})
    return out


def trial_id(overrides, seed, base_seed=42):
    """Folder name of a trial, e.g. "lr3e-04_wd1e-02" or "blocksB9_seed43"."""
    parts = [SHORT.get(k, k) + (f"{v:.0e}" if isinstance(v, float) else str(v))
             for k, v in sorted(flatten(overrides).items())]
    if seed != base_seed:
        parts.append(f"seed{seed}")
    return "_".join(parts) or "base"


def trial_dirs(cfg, exp, tid):
    p = get_paths(cfg)
    ckpt, res = p.checkpoint_dir / exp, p.result_dir / "experiments" / exp
    return (ckpt, res) if tid == "base" else (ckpt / "trials" / tid, res / "trials" / tid)


def run_trial(cfg, exp, overrides, seed, phase, split_df, device, amp):
    tid = trial_id(overrides, seed, cfg["tuning"]["base_seed"])
    ckpt_dir, res_dir = trial_dirs(cfg, exp, tid)
    summary_file = res_dir / f"{exp}_training_summary.json"
    if summary_file.exists():  # written last, so it means the trial finished
        print(f"{exp} {tid}: done, skipped")
        return
    print(f"===== {exp} {tid}")
    res_dir.mkdir(parents=True, exist_ok=True)

    ecfg = merge(cfg["experiments"][exp], overrides)
    classes = load_class_mapping(cfg)["class_name"].tolist()
    kwargs = {"num_classes": len(classes), **{k: ecfg[k] for k in MODEL_ARGS[ecfg["model_name"]]}}
    info = {"experiment_id": exp, "run_id": f"{exp}_{tid}", "trial_id": tid, "phase": phase,
            "overrides": overrides, "seed": seed, "model_name": ecfg["model_name"], "model_kwargs": kwargs,
            "class_names": classes, "split_sha256": cfg["split"]["split_sha256"]}
    loaders = get_dataloaders(merge(cfg, {"project": {"seed": seed}}), split_df=split_df)
    set_seed(seed)

    def train(model, lr, epochs, name):
        opt = build_optimizer([p for p in model.parameters() if p.requires_grad], {**ecfg, "lr": lr})
        return fit(model, *loaders, optimizer=opt, scheduler=build_scheduler(opt, ecfg),
                   criterion=nn.CrossEntropyLoss(), device=device, amp=amp, num_classes=len(classes),
                   max_epochs=epochs, patience=cfg["training"]["early_stopping_patience"],
                   checkpoint_dir=ckpt_dir, experiment_id=name, history_path=res_dir / f"{name}_history.csv",
                   run_info=info)

    if exp == "E3":
        summary = _train_e3(cfg, ecfg, overrides, seed, kwargs, train, ckpt_dir, res_dir, device)
    else:
        model = build_model(ecfg["model_name"], **kwargs).to(device)
        epochs = cfg["training"]["epochs_simple" if exp == "E1" else "epochs_complex"]
        r = train(model, ecfg["lr"], epochs, exp)
        summary = {"best_epoch": r.best_epoch, "best_val_macro_f1": r.best_val_macro_f1,
                   "best_val_metrics": torch.load(r.best_path, map_location="cpu")["val_metrics"],
                   "stopped_early": r.stopped_early, "parameter_count": count_parameters(model),
                   "training_time_minutes": round(r.history["seconds"].sum() / 60, 1)}
    res_dir.mkdir(parents=True, exist_ok=True)  # see train.fit: Drive folders can vanish mid-run
    summary_file.write_text(json.dumps({**info, **summary, "selection_metric": "val_macro_f1"}, indent=2))
    print(f"{exp} {tid}: validation macro-F1 {summary['best_val_macro_f1']:.4f}")


def _train_e3(cfg, ecfg, overrides, seed, kwargs, train, ckpt_dir, res_dir, device):
    model = build_model("EfficientNet-B0", pretrained=True, **kwargs).to(device)

    # stage A (head only) is identical to the base run when only stage B changes, so reuse it
    if set(overrides) <= {"stage_b"} and seed == cfg["tuning"]["base_seed"]:
        base_ckpt, base_res = trial_dirs(cfg, "E3", "base")
        state = torch.load(base_ckpt / "E3_stageA_best.pt", map_location="cpu")
        a = SimpleNamespace(best_path=base_ckpt / "E3_stageA_best.pt", best_epoch=state["epoch"],
                            best_val_macro_f1=round(state["val_metrics"]["macro_f1"], 5),
                            history=pd.read_csv(base_res / "E3_stageA_history.csv"))
    else:
        model.set_trainable_blocks(ecfg["stage_a"]["trainable_blocks"])
        a = train(model, ecfg["stage_a"]["lr"], cfg["training"]["epochs_transfer_head"], "E3_stageA")

    # stage B fine-tunes from the best stage-A weights
    model.load_state_dict(torch.load(a.best_path, map_location=device)["model"])
    model.set_trainable_blocks(ecfg["stage_b"]["trainable_blocks"])
    b = train(model, ecfg["stage_b"]["lr"], cfg["training"]["epochs_finetune"], "E3_stageB")

    best, stage = (b, "B") if b.best_val_macro_f1 > a.best_val_macro_f1 else (a, "A")
    final = torch.load(best.best_path, map_location="cpu")
    torch.save(final, ckpt_dir / "E3_best.pt")
    history = pd.concat([a.history.assign(stage="A"),
                         b.history.assign(stage="B", epoch=b.history["epoch"] + len(a.history))])
    history.to_csv(res_dir / "E3_history.csv", index=False)
    return {"selected_stage": stage, "best_epoch": best.best_epoch + (len(a.history) if stage == "B" else 0),
            "best_val_macro_f1": best.best_val_macro_f1, "best_val_metrics": final["val_metrics"],
            "stopped_early": b.stopped_early, "parameter_count": sum(p.numel() for p in model.parameters()),
            "training_time_minutes": round(history["seconds"].sum() / 60, 1)}


def tuning_table(cfg, exp):
    """One row per finished trial, the base run included."""
    res = trial_dirs(cfg, exp, "base")[1]
    files = [res / f"{exp}_training_summary.json", *sorted((res / "trials").glob(f"*/{exp}_training_summary.json"))]
    rows = []
    for f in files:
        s = json.loads(f.read_text())
        rows.append({"trial_id": s.get("trial_id", "base"), "phase": s.get("phase", 0),
                     "seed": s.get("seed", cfg["tuning"]["base_seed"]),
                     "overrides": json.dumps(s.get("overrides", {}), sort_keys=True),
                     "val_macro_f1": s["best_val_macro_f1"], "best_epoch": s["best_epoch"],
                     "minutes": s["training_time_minutes"]})
    return pd.DataFrame(rows)


def winner(cfg, table, max_phase):
    # best validation macro-F1 among base-seed trials; a tie keeps the earlier phase
    t = table[(table["seed"] == cfg["tuning"]["base_seed"]) & (table["phase"] <= max_phase)]
    return t.sort_values(["val_macro_f1", "phase"], ascending=[False, True], kind="stable").iloc[0]


def plan(cfg, exp, phase):
    """(overrides, seed) of every trial in a phase. Phase 2 and 3 need the previous phase finished."""
    tuning, base_seed = cfg["tuning"], cfg["tuning"]["base_seed"]
    if phase == 1:
        return [(o, base_seed) for o in tuning["phase1"][exp]]
    table = tuning_table(cfg, exp)
    missing = {trial_id(o, s, base_seed) for o, s in plan(cfg, exp, phase - 1)} - set(table["trial_id"])
    if missing:
        raise RuntimeError(f"{exp}: finish phase {phase - 1} first (missing {sorted(missing)})")
    best = json.loads(winner(cfg, table, phase - 1)["overrides"])
    if phase == 2:
        return [(merge(best, o), base_seed) for o in tuning["phase2"][exp]]
    return [(best, s) for s in tuning["phase3_seeds"]]


def write_selection(cfg, exp):
    """After phase 3: save the chosen configuration and its three seed runs for notebook 09."""
    base_seed = cfg["tuning"]["base_seed"]
    table = tuning_table(cfg, exp)
    best = winner(cfg, table, 2)
    overrides = json.loads(best["overrides"])
    seed_trials = {str(s): trial_id(overrides, s, base_seed) for s in [base_seed, *cfg["tuning"]["phase3_seeds"]]}
    missing = set(seed_trials.values()) - set(table["trial_id"])
    if missing:
        raise RuntimeError(f"{exp}: phase 3 not finished (missing {sorted(missing)})")
    scores = table.set_index("trial_id").loc[list(seed_trials.values()), "val_macro_f1"]
    selection = {"selected_trial": best["trial_id"], "overrides": overrides, "seed_trials": seed_trials,
                 "val_macro_f1_mean": round(float(scores.mean()), 4), "val_macro_f1_sd": round(float(scores.std()), 4)}
    res = trial_dirs(cfg, exp, "base")[1]
    (res / "selection.json").write_text(json.dumps(selection, indent=2))
    table.to_csv(res / f"{exp}_seed_runs.csv", index=False)
    return selection
