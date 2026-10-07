import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.classifier_interface import EpochFeatureScorer, SklearnP300Adapter


def test_adapter_returns_binary_logit_and_temperature_scaled_logits():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    model = SklearnP300Adapter(LogisticRegression()).fit(X, y)
    logits = model.decision_function(X)
    assert logits.shape == (4,)
    calibrated = model.calibrate(logits, y)
    assert calibrated.shape == (4,)
    assert model.temperature > 0
    np.testing.assert_allclose(model.calibrated_logits(X), calibrated)


def test_temperature_calibration_improves_nll_on_distorted_scores():
    from src.evaluation.calibration import calibration_metrics

    y = np.array([0, 0, 1, 1, 1, 1, 0, 0] * 10)
    logits = np.array([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0, -2.0] * 10)
    model = SklearnP300Adapter(LogisticRegression())
    before = calibration_metrics(logits, y)
    calibrated = model.calibrate(logits, y)
    after = calibration_metrics(calibrated, y)
    assert model.temperature > 1.0
    assert after["nll"] < before["nll"]


def test_epoch_feature_scorer_applies_locked_window_and_decimation():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(12, 3, 10))
    y = np.array([0, 1] * 6)
    times = -0.1 + np.arange(10) / 10
    window = (times >= 0) & (times <= 0.8 + 1e-8)
    features = X[:, :, window][:, :, ::2].reshape(12, -1)
    scaler = StandardScaler().fit(features)
    estimator = LogisticRegression().fit(scaler.transform(features), y)
    scorer = EpochFeatureScorer(
        estimator, scaler, ["Fz", "Cz", "Pz"], 10, -0.1,
        {"window_s": 0.8, "decim": 2}, temperature=2.0,
    )
    logits = scorer.calibrated_logits(X)
    assert logits.shape == (12,)
    np.testing.assert_allclose(logits, scorer.decision_function(X) / 2)
