"""Metrics, reporting, and calibration for all models."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)

logger = logging.getLogger(__name__)

RESULT_LABELS = {0: "Away Win", 1: "Draw", 2: "Home Win"}


def evaluate_classifier(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label: str = "test",
) -> dict:
    """Compute classification metrics. Returns dict suitable for experiment_log.json."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    ll = log_loss(y_true, y_proba, labels=[0, 1, 2])

    # Brier score: per-class OvR average
    brier_scores = []
    for cls in [0, 1, 2]:
        bs = brier_score_loss((y_true == cls).astype(int), y_proba[:, cls])
        brier_scores.append(bs)
    brier = float(np.mean(brier_scores))

    # Naive baseline: always predict home win (class 2)
    baseline_pred = np.full(len(y_true), 2)
    baseline_acc = accuracy_score(y_true, baseline_pred)

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "brier_score": round(brier, 4),
        "baseline_accuracy": round(baseline_acc, 4),
        "accuracy_vs_baseline": round(acc - baseline_acc, 4),
    }

    logger.info(
        "[%s] accuracy=%.4f (baseline=%.4f, Δ=%.4f)  f1_macro=%.4f  logloss=%.4f  brier=%.4f",
        label, acc, baseline_acc, acc - baseline_acc, f1, ll, brier,
    )
    _log_class_breakdown(y_true, y_pred)
    return metrics


def evaluate_regressor(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label: str = "test",
) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    naive_mae = float(np.mean(np.abs(y_true - y_true.mean())))
    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "naive_mae": round(naive_mae, 4),
    }
    logger.info("[%s] MAE=%.4f (naive=%.4f)  RMSE=%.4f", label, mae, naive_mae, rmse)
    return metrics


def generate_calibration_plot(
    y_true: pd.Series,
    y_proba: np.ndarray,
    out_path: Path,
    model_label: str = "XGB Classifier",
) -> None:
    """Save calibration curve plot for each outcome class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Calibration Curves — {model_label}", fontsize=13)

    for cls, ax in zip([0, 1, 2], axes):
        y_bin = (y_true == cls).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, cls], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o", label=model_label)
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.set_title(RESULT_LABELS[cls])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Calibration plot saved → %s", out_path)


def _log_class_breakdown(y_true: pd.Series, y_pred: np.ndarray) -> None:
    for cls in [0, 1, 2]:
        mask = y_true == cls
        if mask.sum() == 0:
            continue
        acc_cls = accuracy_score(y_true[mask], y_pred[mask])
        logger.info("  %-10s | n=%4d | accuracy=%.3f", RESULT_LABELS[cls], mask.sum(), acc_cls)
