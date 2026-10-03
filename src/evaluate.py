"""One shared evaluation implementation for every experiment (CONSTITUTION C2, EXPERIMENT_PROTOCOL).

Used on the validation set during development and, once per frozen checkpoint, on the test set.
Macro averages are taken over all K classes (`labels=range(K)`, zero_division=0).
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@dataclass
class Predictions:
    filepaths: list
    y_true: np.ndarray
    y_pred: np.ndarray
    probs: np.ndarray
    loss: float | None


@torch.no_grad()
def predict(model, loader, device, criterion=None, amp=False):
    """Run the model over a loader in eval mode; `loss` is the mean loss when `criterion` is given."""
    model.eval()
    probs, labels, filepaths = [], [], []
    total_loss = 0.0
    for images, targets, paths in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, targets).item() * targets.size(0)
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(targets.cpu())
        filepaths.extend(paths)
    probs = torch.cat(probs).numpy()
    y_true = torch.cat(labels).numpy()
    loss = total_loss / len(y_true) if criterion is not None else None
    return Predictions(filepaths, y_true, probs.argmax(axis=1), probs, loss)


def classification_metrics(y_true, y_pred, num_classes):
    """Accuracy and macro precision/recall/F1 over all classes, as plain floats."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def per_class_metrics(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    return pd.DataFrame({
        "class_index": range(len(class_names)),
        "class_name": class_names,
        "support": support,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
    })


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    """Counts in each cell; colour is the row-normalised share (recall per true class)."""
    counts = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    shares = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=8,
                    color="white" if shares[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig
