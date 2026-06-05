from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from dataset import CLASS_NAMES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str] = CLASS_NAMES) -> dict[str, float]:
    del class_names
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "uar": float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_path: str | Path,
    class_names: Sequence[str] = CLASS_NAMES,
    title: str = "Confusion Matrix",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    figure, axis = plt.subplots(figsize=(7, 5.5), dpi=140)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
