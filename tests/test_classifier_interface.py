import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models.classifier_interface import SklearnP300Adapter


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
