"""Stepwise linear discriminant feature selection for P300 epochs."""

from __future__ import annotations

import numpy as np
from scipy.stats import t as student_t
from typing import Optional


class StepwiseLinearDiscriminant:
    """Weighted forward/backward stepwise regression with p-value cutoffs.

    This implements the conventional SWLDA idea: add the most significant
    unused feature below ``p_enter`` and remove selected features above
    ``p_remove``. The returned score is a linear target-vs-nontarget logit
    surrogate; temperature calibration is handled by the classifier wrapper.
    """

    def __init__(
        self, p_enter: float = 0.05, p_remove: float = 0.10,
        max_features: int = 35, class_weight: Optional[dict[int, float]] = None,
    ):
        if not 0 < p_enter < p_remove < 1:
            raise ValueError("Require 0 < p_enter < p_remove < 1")
        self.p_enter = float(p_enter)
        self.p_remove = float(p_remove)
        self.max_features = int(max_features)
        self.class_weight = class_weight or {0: 1.0, 1: 7.0}
        self.selected_features_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def _fit_selected(self, X: np.ndarray, y: np.ndarray, selected: list[int], sqrt_w: np.ndarray):
        design = np.column_stack([np.ones(len(y)), X[:, selected]])
        design_w = design * sqrt_w[:, None]
        target_w = y * sqrt_w
        coef = np.linalg.lstsq(design_w, target_w, rcond=None)[0]
        residual = target_w - design_w @ coef
        df = max(1, len(y) - design_w.shape[1])
        mse = float(residual @ residual / df)
        covariance = np.linalg.pinv(design_w.T @ design_w)
        se = np.sqrt(np.maximum(0.0, np.diag(covariance) * mse))
        t_values = np.divide(coef, se, out=np.full_like(coef, np.inf), where=se > 1e-12)
        p_values = 2.0 * student_t.sf(np.abs(t_values), df)
        return coef, residual, mse, design_w, p_values

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2D and aligned with y")
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("SWLDA requires both binary classes (0 and 1)")
        weights = np.where(y == 1, self.class_weight[1], self.class_weight[0])
        sqrt_w = np.sqrt(weights)
        Xw = X * sqrt_w[:, None]
        yw = y * sqrt_w
        selected: list[int] = []

        for _ in range(min(self.max_features, X.shape[1])):
            coef, residual, mse, design_w, _ = self._fit_selected(X, y, selected, sqrt_w)
            Q, _ = np.linalg.qr(design_w, mode="reduced")
            projected = Q.T @ Xw
            norms = np.maximum(1e-12, np.sum(Xw * Xw, axis=0) - np.sum(projected * projected, axis=0))
            cov = Xw.T @ residual
            df = max(1, len(y) - design_w.shape[1] - 1)
            t_values = cov / np.sqrt(np.maximum(1e-12, mse * norms))
            p_values = 2.0 * student_t.sf(np.abs(t_values), df)
            if selected:
                p_values[np.asarray(selected)] = np.inf
            candidate = int(np.argmin(p_values))
            if float(p_values[candidate]) >= self.p_enter:
                if not selected:
                    selected = [candidate]
                else:
                    break
            else:
                selected.append(candidate)

            # Backward step: remove the least significant selected term until
            # all remaining terms satisfy the configured removal threshold.
            while selected:
                coef, _, _, _, selected_p = self._fit_selected(X, y, selected, sqrt_w)
                removable = selected_p[1:]
                if not len(removable) or float(np.max(removable)) <= self.p_remove:
                    break
                del selected[int(np.argmax(removable))]

        if not selected:
            raise RuntimeError("SWLDA selected no features; relax p_enter or inspect the fold")
        coef, _, _, _, _ = self._fit_selected(X, y, selected, sqrt_w)
        self.selected_features_ = np.asarray(selected, dtype=int)
        self.intercept_ = float(coef[0])
        self.coef_ = np.asarray(coef[1:], dtype=np.float64)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None or self.coef_ is None:
            raise RuntimeError("Call fit before decision_function")
        X = np.asarray(X)
        return self.intercept_ + X[:, self.selected_features_] @ self.coef_
