"""Classification metrics. Primary metric is macro F1."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from workbench.config import LABELS, PRIMARY_METRIC


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = LABELS,
) -> dict[str, float]:
    return {
        PRIMARY_METRIC: float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def per_class_report(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...] = LABELS,
) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        output_dict=True,
        zero_division=0,
    )
