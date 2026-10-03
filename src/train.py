"""Shared training loop: validation-only model selection, early stopping, Drive checkpoints, resume.

Selection and early stopping both use validation macro F1 (DECISION_LOG D010). The test set is
never touched here. Checkpoints are written every epoch so a Colab disconnect costs at most one
epoch; a resumed run continues the same schedule but is not bit-identical to an uninterrupted one.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.evaluate import classification_metrics, predict


@dataclass
class FitResult:
    history: pd.DataFrame
    best_epoch: int
    best_val_macro_f1: float
    stopped_early: bool
    best_path: Path
    last_path: Path
    run_info: dict


def build_optimizer(parameters, exp_cfg):
    if exp_cfg["optimizer"] != "adamw":
        raise ValueError(f"Unsupported optimizer: {exp_cfg['optimizer']}")
    return torch.optim.AdamW(parameters, lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])


def build_scheduler(optimizer, exp_cfg):
    if exp_cfg["scheduler"] is None:
        return None
    if exp_cfg["scheduler"] != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler: {exp_cfg['scheduler']}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=exp_cfg["scheduler_factor"], patience=exp_cfg["scheduler_patience"]
    )


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp):
    """Mean training loss and accuracy over the epoch (augmented images, dropout active)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, *, optimizer, criterion, device, num_classes, max_epochs, patience,
        checkpoint_dir, experiment_id, history_path, run_info, scheduler=None, amp=False, resume=True):
    """Train until `max_epochs` or until validation macro F1 has not improved for `patience` epochs.

    Writes `{id}_best.pt` (best validation macro F1; ties keep the earlier epoch), `{id}_last.pt`
    (full state for resuming) and the history CSV after every epoch. If `{id}_last.pt` exists and
    `resume` is True, training continues from it; a finished run is returned without retraining.
    `run_info` (model name, run id, ...) is stored in both checkpoints; on resume the stored copy wins.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(history_path).parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{experiment_id}_best.pt"
    last_path = checkpoint_dir / f"{experiment_id}_last.pt"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history, best_f1, best_epoch, stale, start_epoch = [], -1.0, 0, 0, 1
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_f1, best_epoch = state["history"], state["best_val_macro_f1"], state["best_epoch"]
        stale, run_info = state["epochs_without_improvement"], state["run_info"]
        start_epoch = state["epoch"] + 1
        if state["finished"]:
            print(f"{experiment_id} already finished at epoch {state['epoch']}; nothing to train.")
            return FitResult(pd.DataFrame(history), best_epoch, best_f1, stale >= patience, best_path, last_path, run_info)
        print(f"Resuming {experiment_id} ({run_info['run_id']}) at epoch {start_epoch}")

    for epoch in range(start_epoch, max_epochs + 1):
        started = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
        val = predict(model, val_loader, device, criterion, amp)
        val_metrics = classification_metrics(val.y_true, val.y_pred, num_classes)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_accuracy": round(train_acc, 5),
            "val_loss": round(val.loss, 5),
            "val_accuracy": round(val_metrics["accuracy"], 5),
            "val_macro_f1": round(val_metrics["macro_f1"], 5),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)

        if row["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = row["val_macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics,
                        "val_loss": val.loss, "run_info": run_info}, best_path)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(row["val_macro_f1"])

        finished = stale >= patience or epoch == max_epochs
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_val_macro_f1": best_f1,
            "best_epoch": best_epoch,
            "epochs_without_improvement": stale,
            "finished": finished,
            "run_info": run_info,
        }, last_path)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(f"epoch {epoch:>3} | train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
              f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f} macro-F1 {row['val_macro_f1']:.4f} | "
              f"lr {row['lr']:.2e} | {row['seconds']:.0f}s" + ("  *best*" if best_epoch == epoch else ""))
        if finished:
            break

    stopped_early = stale >= patience
    if stopped_early:
        print(f"Early stopping: no validation macro-F1 improvement for {patience} epochs.")
    print(f"Best epoch {best_epoch}: validation macro-F1 {best_f1:.4f}")
    return FitResult(pd.DataFrame(history), best_epoch, best_f1, stopped_early, best_path, last_path, run_info)


def plot_history(history, title):
    """Training/validation loss and accuracy per epoch (report figure)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_ylabel("cross-entropy loss")
    axes[1].plot(history["epoch"], history["train_accuracy"], label="train")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="validation")
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="validation macro-F1", linestyle="--")
    axes[1].set_ylabel("score")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
