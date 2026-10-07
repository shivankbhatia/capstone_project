import numpy as np

from scripts.run_clean_m0_ablation_ladder import _condition, _sequence_evidence, _aggregate


def test_ladder_drops_partial_sequences_without_both_row_and_column_evidence():
    assert _sequence_evidence(np.array([1.0, 2.0]), np.array([1, 2])) is None
    evidence = _sequence_evidence(np.array([1.0, 2.0, 3.0]), np.array([1, 10, 10]))
    assert evidence is not None
    assert evidence["valid_flash_count"] == 3
    assert np.isclose(evidence["grid"].sum(), 1.0)


def test_ladder_condition_groups_dyn_variants_together():
    assert _condition("D_01_SE001_RC_Train01") == "RC/Train"
    assert _condition("D_01_SE001_Dyn_Test01") == "Dyn"
    assert _condition("D_01_SE001_DynBigram_Test01") == "DynBigram"


def test_aggregate_reports_zero_coverage_and_uses_recorded_flash_count():
    records = [
        {"rung": 0, "condition": "RC/Train", "subject": "D_01", "correct": True,
         "zero_coverage": False, "flashes_used": 34, "sequences_used": 2,
         "time_seconds": 4.0, "sequence_nll": 1.0},
        {"rung": 0, "condition": "RC/Train", "subject": "D_01", "correct": None,
         "zero_coverage": True, "flashes_used": 0, "sequences_used": 0,
         "time_seconds": 0.0, "sequence_nll": None},
    ]
    result = _aggregate(records, 0, "RC/Train")
    assert result["characters"] == 1
    assert result["zero_coverage_terminal_characters"] == 1
    assert result["flashes_per_character"] == 34.0
    assert result["mean_time_seconds"] == 4.0
