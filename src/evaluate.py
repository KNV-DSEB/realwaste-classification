from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def predict(model, loader, device, criterion=None, amp=False):
    model.eval()
    probs, labels, filepaths, total_loss = [], [], [], 0.0
    for images, targets, paths in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(images).float()
        if criterion is not None:
            total_loss += criterion(logits, targets).item() * len(targets)
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(targets.cpu())
        filepaths += paths
    probs, y_true = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    return SimpleNamespace(filepaths=filepaths, y_true=y_true, y_pred=probs.argmax(1), probs=probs,
                           loss=total_loss / len(y_true) if criterion is not None else None)


def classification_metrics(y_true, y_pred, num_classes):
    # macro = plain average over all classes, so small classes count as much as big ones
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes),
                                                  average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


def per_class_metrics(y_true, y_pred, class_names):
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)),
                                                        average=None, zero_division=0)
    return pd.DataFrame({"class_index": range(len(class_names)), "class_name": class_names, "support": support,
                         "precision": p.round(4), "recall": r.round(4), "f1": f1.round(4)})


def plot_confusion_matrix(y_true, y_pred, class_names, title):
    counts = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)  # colour = recall of each true class
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set(xlabel="Predicted class", ylabel="True class", title=title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, counts[i, j], ha="center", va="center", fontsize=8,
                    color="white" if share[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig
