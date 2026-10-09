import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.evaluate import classification_metrics, predict


def build_optimizer(params, exp_cfg):
    return torch.optim.AdamW(params, lr=exp_cfg["lr"], weight_decay=exp_cfg["weight_decay"])


def build_scheduler(optimizer, exp_cfg):
    # halve the learning rate when validation macro-F1 stops improving
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=exp_cfg["scheduler_factor"], patience=exp_cfg["scheduler_patience"])


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=amp):  # mixed precision on GPU
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(targets)
        correct += (logits.argmax(1) == targets).sum().item()
        seen += len(targets)
    return total_loss / seen, correct / seen


def fit(model, train_loader, val_loader, *, optimizer, criterion, device, num_classes, max_epochs, patience,
        checkpoint_dir, experiment_id, history_path, run_info, scheduler=None, amp=False):
    """Train with early stopping on validation macro-F1.

    Saves {id}_best.pt (best epoch) and {id}_last.pt (everything needed to resume) after each epoch.
    If {id}_last.pt exists, training continues from it, so a Colab disconnect only loses one epoch.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{experiment_id}_best.pt"
    last_path = checkpoint_dir / f"{experiment_id}_last.pt"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history, best_f1, best_epoch, stale, start = [], -1.0, 0, 0, 1

    if last_path.exists():
        state = torch.load(last_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, best_f1, best_epoch = state["history"], state["best_val_macro_f1"], state["best_epoch"]
        stale, run_info, start = state["epochs_without_improvement"], state["run_info"], state["epoch"] + 1
        if stale >= patience or state["epoch"] >= max_epochs:
            print(f"{experiment_id} already finished at epoch {state['epoch']}")
            return _result(history, best_epoch, best_f1, stale >= patience, best_path, last_path, run_info)
        print(f"Resuming {experiment_id} at epoch {start}")

    for epoch in range(start, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp)
        val = predict(model, val_loader, device, criterion, amp)
        metrics = classification_metrics(val.y_true, val.y_pred, num_classes)
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "train_accuracy": round(train_acc, 5),
               "val_loss": round(val.loss, 5), "val_accuracy": round(metrics["accuracy"], 5),
               "val_macro_f1": round(metrics["macro_f1"], 5), "lr": optimizer.param_groups[0]["lr"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        if row["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = row["val_macro_f1"], epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_metrics": metrics,
                        "run_info": run_info}, best_path)
        else:
            stale += 1
        if scheduler:
            scheduler.step(row["val_macro_f1"])

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict(),
                    "epoch": epoch, "history": history, "best_val_macro_f1": best_f1, "best_epoch": best_epoch,
                    "epochs_without_improvement": stale, "run_info": run_info}, last_path)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"epoch {epoch:>3} | train {train_loss:.4f} / {train_acc:.4f} | val {val.loss:.4f} / "
              f"{metrics['accuracy']:.4f} / F1 {metrics['macro_f1']:.4f} | {row['seconds']:.0f}s"
              + ("  *best*" if best_epoch == epoch else ""))
        if stale >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    print(f"Best epoch {best_epoch}: validation macro-F1 {best_f1:.4f}")
    return _result(history, best_epoch, best_f1, stale >= patience, best_path, last_path, run_info)


def _result(history, best_epoch, best_f1, stopped_early, best_path, last_path, run_info):
    return SimpleNamespace(history=pd.DataFrame(history), best_epoch=best_epoch, best_val_macro_f1=best_f1,
                           stopped_early=stopped_early, best_path=best_path, last_path=last_path, run_info=run_info)


def plot_history(history, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train")
    ax1.plot(history["epoch"], history["val_loss"], label="validation")
    ax1.set_ylabel("loss")
    ax2.plot(history["epoch"], history["train_accuracy"], label="train")
    ax2.plot(history["epoch"], history["val_accuracy"], label="validation")
    ax2.plot(history["epoch"], history["val_macro_f1"], "--", label="validation macro-F1")
    ax2.set_ylabel("score")
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig
