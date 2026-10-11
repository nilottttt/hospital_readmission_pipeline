"""Evaluation metrics, threshold selection, calibration and plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted average |observed rate - mean predicted probability| over
    equal-frequency bins (quantile bins work better for rare outcomes)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    if len(y_true) == 0:
        return float("nan")
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(y_true.mean() - y_prob.mean()))
    idx = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def best_f1_threshold(y_true, y_prob) -> float:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)[:-1]
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def classification_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        "flag_rate": float(y_pred.mean()),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def subgroup_metrics(df: pd.DataFrame, y_col: str, p_col: str, threshold: float,
                     group_cols: list[str], min_n: int = 50) -> pd.DataFrame:
    rows = []
    for g in group_cols:
        for value, part in df.groupby(g, observed=True):
            if len(part) < min_n:
                continue
            y, p = part[y_col].to_numpy(), part[p_col].to_numpy()
            pred = (p >= threshold).astype(int)
            rows.append({
                "group_type": g, "group_value": str(value), "n": len(part),
                "positives": int(y.sum()),
                "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
                "recall": float(recall_score(y, pred, zero_division=0)),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "flag_rate": float(pred.mean()),
                "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# plots (saved as PNG and logged to MLflow)
# ----------------------------------------------------------------------------
def _finish(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_pr(curves: dict, out_dir: Path) -> list[Path]:
    """curves = {model_name: (y_true, y_prob)}"""
    colors = [BLUE, ORANGE, AQUA, "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for (name, (y, p)), c in zip(curves.items(), colors):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(frameon=False, fontsize=8)
    roc = _finish(fig, out_dir / "roc_curve.png")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for (name, (y, p)), c in zip(curves.items(), colors):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=c, lw=2, label=f"{name} (AP {average_precision_score(y, p):.3f})")
    base = np.mean(next(iter(curves.values()))[0])
    ax.axhline(base, color="grey", lw=1, ls="--", label=f"base rate {base:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test set)")
    ax.legend(frameon=False, fontsize=8)
    pr = _finish(fig, out_dir / "pr_curve.png")
    return [roc, pr]


def plot_calibration(y_true, probs: dict, out_dir: Path, name="calibration_curve.png") -> Path:
    """probs = {label: y_prob}"""
    colors = [ORANGE, BLUE, AQUA]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
    for (label, p), c in zip(probs.items(), colors):
        frac, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=5, lw=2, color=c,
                label=f"{label} (ECE {expected_calibration_error(y_true, p):.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed 30-day readmission rate",
           title="Calibration (test set)")
    lim = max(0.5, max(np.max(p) for p in probs.values()))
    ax.set_xlim(0, min(1, lim)); ax.set_ylim(0, min(1, lim))
    ax.legend(frameon=False, fontsize=8)
    return _finish(fig, out_dir / name)


def plot_confusion(m: dict, out_dir: Path) -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
    ax.set_yticks([0, 1], ["Actual no", "Actual yes"])
    ax.set_title("Confusion matrix (test, chosen threshold)")
    return _finish(fig, out_dir / "confusion_matrix.png")


def plot_importance(imp: pd.Series, title: str, path: Path, top: int = 15) -> Path:
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(6, 0.32 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return _finish(fig, path)
