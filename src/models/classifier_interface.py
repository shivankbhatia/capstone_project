"""Common calibrated per-flash score interface for P300 classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit


class P300Classifier(ABC):
    """Classifier API shared by classical and neural P300 models.

    ``decision_function`` returns the target-vs-nontarget logit. Temperature
    fitting is deliberately separate so callers can supply inner-fold OOF
    logits without fitting calibration on the model's training data.
    """

    def __init__(self) -> None:
        self.temperature = 1.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups=None):
        raise NotImplementedError

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, logits: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit scalar temperature on calibration logits and return scaled logits."""
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if logits.shape != y.shape or logits.size == 0:
            raise ValueError("logits and y must be non-empty arrays of equal length")
        if not np.isin(y, [0, 1]).all():
            raise ValueError("Temperature calibration requires binary labels 0/1")

        def nll(log_temperature: float) -> float:
            scaled = logits / np.exp(log_temperature)
            # Stable binary cross-entropy from logits.
            return float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))

        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature = float(np.exp(result.x))
        return logits / self.temperature

    def calibrated_logits(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.decision_function(X), dtype=np.float64).reshape(-1) / self.temperature

    def calibrated_probabilities(self, X: np.ndarray) -> np.ndarray:
        return expit(self.calibrated_logits(X))


class SklearnP300Adapter(P300Classifier):
    """Adapt a sklearn estimator or pipeline to the common score interface."""

    def __init__(self, estimator):
        super().__init__()
        self.estimator = estimator

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None):
        self.estimator.fit(X, y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "decision_function"):
            scores = np.asarray(self.estimator.decision_function(X))
            if scores.ndim == 2:
                if scores.shape[1] != 1:
                    raise ValueError("Expected a binary estimator with one decision score")
                scores = scores[:, 0]
            return scores.reshape(-1)
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError("Estimator must expose decision_function or predict_proba")
        probabilities = np.clip(self.estimator.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(probabilities) - np.log1p(-probabilities)


class EpochFeatureScorer:
    """Serializable inference wrapper for a trained epoch-level estimator.

    It applies the locked time window and decimation before scoring. The input
    channel order must match the saved channel list from the clean epoch cache.
    """

    def __init__(self, estimator, scaler, channels, sfreq, tmin, config, temperature=1.0):
        self.estimator = estimator
        self.scaler = scaler
        self.channels = tuple(channels)
        self.sfreq = float(sfreq)
        self.tmin = float(tmin)
        self.config = dict(config)
        self.temperature = float(temperature)

    def _features(self, X):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Expected epochs shaped (n_epochs, n_channels, n_times)")
        if X.shape[1] != len(self.channels):
            raise ValueError(
                f"Expected {len(self.channels)} channels in locked order, received {X.shape[1]}"
            )
        times = self.tmin + np.arange(X.shape[-1]) / self.sfreq
        keep = (times >= 0.0) & (times <= float(self.config["window_s"]) + 1e-8)
        features = X[:, :, keep][:, :, ::int(self.config["decim"])].reshape(len(X), -1)
        return features

    def decision_function(self, X):
        return np.asarray(self.estimator.decision_function(
            self.scaler.transform(self._features(X))
        )).reshape(-1)

    def calibrated_logits(self, X):
        return self.decision_function(X) / self.temperature
