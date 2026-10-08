import json
from pathlib import Path

from scripts.separate_classifier_label_vault import partition_registry


def test_registry_partition_seals_manifest_and_non_training_sessions():
    current = {
        "D_01_SE001_RC_Train01": "A",
        "D_01_SE001_RC_Train06": "B",
        "D_01_SE001_Dyn_Test01": "C",
        "E_01_SE001_RC_Train01": "D",
    }
    visible, vault = partition_registry(
        current, {"old evaluation entry": "KEEP"},
        {"D_01_SE001_RC_Train06"},
    )
    assert set(visible) == {"D_01_SE001_RC_Train01"}
    assert set(vault) == {
        "old evaluation entry", "D_01_SE001_RC_Train06",
        "D_01_SE001_Dyn_Test01", "E_01_SE001_RC_Train01",
    }
    assert not set(visible) & set(vault)


def test_classifier_training_sources_cannot_reference_label_vault():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "scripts/build_classifier_cache.py",
        root / "scripts/build_study_q_classifier_cache.py",
        root / "scripts/screen_shrinkage_lda.py",
        root / "scripts/run_classical_cv.py",
        root / "scripts/run_study_q_classifier_cv.py",
        root / "scripts/nested_study_q_calibration_fusion.py",
        root / "scripts/summarize_study_q_oof.py",
        root / "scripts/build_study_q_phrase_banks.py",
        root / "scripts/run_riemann_cv.py",
        root / "scripts/run_eegnet_cv.py",
        root / "scripts/nested_sequence_calibration.py",
        root / "src/models/classifier_interface.py",
        root / "src/models/eegnet.py",
    ]
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert "ground_truth_vault.json" not in source
        assert "include_heldout=True" not in source


def test_study_q_test_evaluator_train_dry_run_does_not_read_test_labels():
    from scripts.evaluate_study_q_test_once import run
    import pytest

    result = run(allow_heldout_eval=False, dry_run_train=True)
    assert result == {"dry_run": True, "manifest_verified": True, "test_labels_read": False}
    with pytest.raises(PermissionError):
        run(allow_heldout_eval=False, dry_run_train=False)


def test_study_q_training_registry_excludes_test_and_same_session_labels():
    root = Path(__file__).resolve().parents[1]
    train_registry = json.loads((root / "data/processed/study_q_train_registry.json").read_text())
    vault = json.loads((root / "data/evaluation/ground_truth_vault.json").read_text())
    manifest = json.loads((root / "splits/study_q_manifest.json").read_text())
    excluded = json.loads((root / "splits/study_q_excluded_training_runs.json").read_text())
    heldout = {run for runs in manifest.values() for run in runs}
    excluded_train = {run for runs in excluded.values() for run in runs}
    q_root = root / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
    all_test_runs = {
        p.stem for p in q_root.rglob("*.edf") if "/Test/" in p.as_posix()
    }
    expected_train = {
        p.stem for p in q_root.rglob("*.edf")
        if "/Train/" in p.as_posix() and "_SE003_" not in p.stem
    }
    assert set(train_registry) == expected_train
    assert not (set(train_registry) & (heldout | excluded_train))
    assert not (set(train_registry) & vault.keys())
    assert all_test_runs <= vault.keys()
    assert not (all_test_runs & train_registry.keys())
    global_registry = json.loads((root / "data/processed/ground_truth_registry.json").read_text())
    assert not any(run.startswith("Q_") for run in global_registry)


def test_sequence_trial_loader_keeps_vault_opt_in(tmp_path):
    from src.preprocessing.build_session_sequence import yield_character_trials

    training = tmp_path / "training.json"
    vault = tmp_path / "evaluation.json"
    training.write_text(json.dumps({"D_01_SE001_RC_Train01": "AB"}))
    vault.write_text(json.dumps({"D_01_SE001_Dyn_Test01": "CD"}))
    default_trials = list(yield_character_trials(
        str(training), study="StudyD", evaluation_registry_path=str(vault)
    ))
    assert len(default_trials) == 2
    assert {trial["session_id"] for trial in default_trials} == {"D_01_SE001_RC_Train01"}
    eval_trials = list(yield_character_trials(
        str(training), study="StudyD", include_heldout=True,
        evaluation_registry_path=str(vault),
    ))
    assert len(eval_trials) == 4
    assert {trial["session_id"] for trial in eval_trials} == {
        "D_01_SE001_RC_Train01", "D_01_SE001_Dyn_Test01"
    }
