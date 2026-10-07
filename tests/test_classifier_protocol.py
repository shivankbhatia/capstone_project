import pytest

from src.evaluation.classifier_protocol import assert_training_pool_disjoint


def test_training_pool_rejects_manifest_run():
    with pytest.raises(ValueError, match="held-out run"):
        assert_training_pool_disjoint(
            ["D_01_SE001_RC_Train06-epo.fif"],
            {"D_01_SE001_RC_Train06"},
        )


def test_training_pool_rejects_test_label_even_if_not_in_manifest():
    with pytest.raises(ValueError, match="Test-labelled run"):
        assert_training_pool_disjoint(
            ["D_01_SE001_Dyn_Test01-epo.fif"], set()
        )


def test_training_pool_accepts_clean_calibration_run():
    assert_training_pool_disjoint(
        ["D_01_SE001_RC_Train01-epo.fif"],
        {"D_01_SE001_Dyn_Test01"},
    )
