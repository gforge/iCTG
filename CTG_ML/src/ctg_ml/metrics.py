from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    thresholds = np.unique(np.clip(prob, 0.0, 1.0))
    if len(thresholds) > 500:
        thresholds = np.linspace(0.0, 1.0, 501)
    best_thr = 0.5
    best_score = -1.0
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr
