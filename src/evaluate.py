"""Evaluation helpers for training reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def save_training_history(history: Any, output_dir: str | Path) -> None:
    """Save training metrics and an accuracy plot."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = {key: [float(value) for value in values] for key, values in history.history.items()}
    (output_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="accuracy")
    plt.plot(history.history.get("val_accuracy", []), label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig(output_path / "training_accuracy.png", bbox_inches="tight")
    plt.close()


def save_classification_report(
    model: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    output_dir: str | Path,
) -> None:
    """Save classification report and confusion matrix as JSON artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    class_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [
        labels[class_id] if class_id < len(labels) else f"Class_{class_id}"
        for class_id in class_ids
    ]
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids).tolist()

    (output_path / "classification_report.json").write_text(json.dumps(report, indent=2))
    (output_path / "confusion_matrix.json").write_text(json.dumps(matrix, indent=2))
