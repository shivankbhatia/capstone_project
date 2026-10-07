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
        root / "scripts/screen_shrinkage_lda.py",
        root / "scripts/run_classical_cv.py",
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
