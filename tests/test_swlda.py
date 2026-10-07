import numpy as np

from src.models.swlda import StepwiseLinearDiscriminant


def test_stepwise_lda_selects_signal_feature_and_scores_binary_epochs():
    rng = np.random.default_rng(19)
    y = np.tile([0, 1], 250)
    X = rng.normal(size=(len(y), 12))
    X[:, 4] += 1.5 * (2 * y - 1)
    model = StepwiseLinearDiscriminant(
        p_enter=0.05, p_remove=0.10, max_features=6,
    ).fit(X, y)
    assert 4 in model.selected_features_
    assert model.decision_function(X).shape == y.shape
    assert np.mean(model.decision_function(X)[y == 1]) > np.mean(
        model.decision_function(X)[y == 0]
    )
