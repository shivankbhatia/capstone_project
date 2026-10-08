import json
from pathlib import Path


def test_study_q_audit_matches_frozen_split_and_flags_stopping_censoring():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "results/tables/study_q_audit.json").read_text())
    manifest = json.loads((root / "splits/study_q_manifest.json").read_text())
    assert audit["manifest_run_count"] == sum(map(len, manifest.values())) == 180
    extraction = audit["ground_truth_extraction"]
    assert extraction["runs_compared_with_vault"] == 180
    assert extraction["exact_matches"] == 180
    assert not extraction["mismatches"] and not extraction["errors"]
    geometry, = audit["recording_geometry"]
    assert geometry["grid"] == [9, 8]
    assert geometry["channel_count"] == 32
    assert geometry["flashes_per_sequence"] == 12
    assert audit["stopping_depth_coverage"]["right_censored_for_max_10_replay"]
    assert audit["stopping_depth_coverage"]["characters_with_fewer_than_max_sequences_recorded"] > 0
