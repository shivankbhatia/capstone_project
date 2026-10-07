"""Calibration metrics for binary per-flash target logits."""

from __future__ import annotations

import numpy as np
from scipy.special import expit
from sklearn.metrics import log_loss


def calibration_metrics(logits, y, n_bins: int = 15) -> dict[str, float]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.uint8).reshape(-1)
    if logits.shape != y.shape or logits.size == 0:
        raise ValueError("logits and labels must be non-empty and have equal length")
    if not np.isfinite(logits).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("logits must be finite and labels must be binary")
    p = expit(np.clip(logits, -60, 60))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (p >= low) & (p < high if high < 1 else p <= high)
        if in_bin.any():
            ece += float(in_bin.mean()) * abs(float(p[in_bin].mean()) - float(y[in_bin].mean()))
    return {"nll": float(log_loss(y, p, labels=[0, 1])), "ece": float(ece)}
