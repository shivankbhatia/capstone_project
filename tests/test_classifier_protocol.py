import pytest
import json
from pathlib import Path

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


def test_study_q_session_manifest_and_exclusions_are_disjoint():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "splits/study_q_manifest.json").read_text())
    heldout = {run for runs in manifest.values() for run in runs}
    excluded = json.loads((root / "splits/study_q_excluded_training_runs.json").read_text())
    excluded_train = {run for runs in excluded.values() for run in runs}
    q_root = root / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
    test_runs = {p.stem for p in q_root.rglob("*.edf") if "/SE003/Test/" in p.as_posix()}
    train_runs = {p.stem for p in q_root.rglob("*.edf") if "/SE003/Train/" in p.as_posix()}
    assert heldout == test_runs
    assert excluded_train == train_runs
    assert not (heldout & excluded_train)
    assert all("_Test" in run for run in heldout)
    assert all("_Train" in run for run in excluded_train)


def test_study_q_test_targets_are_vault_only():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "splits/study_q_manifest.json").read_text())
    heldout = {run for runs in manifest.values() for run in runs}
    visible = json.loads((root / "data/processed/ground_truth_registry.json").read_text())
    vault = json.loads((root / "data/evaluation/ground_truth_vault.json").read_text())
    assert not (heldout & visible.keys())
    assert heldout <= vault.keys()
    assert not any(run.startswith("Q_") for run in visible)
