#!/usr/bin/env python3
"""Freeze and audit the classifier experiment's run-level split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys

import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.classifier_protocol import (
    assert_training_pool_disjoint,
    load_heldout_run_ids,
    run_id_from_epoch_path,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/processed/test_sessions_by_subject.json"
FROZEN_MANIFEST = ROOT / "splits/heldout_manifest.json"
RESULT_PATH = ROOT / "results/tables/classifier_protocol_audit.json"
EPOCH_DIR = ROOT / "data/processed/StudyD"
CNN_METADATA = ROOT / "data/processed/cnn_ranked_metadata.json"


def condition(run_id: str) -> str:
    if "DynBigram" in run_id:
        return "DynBigram"
    if "Dyn" in run_id:
        return "Dyn"
    return "RC/Train"


def main() -> None:
    FROZEN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_MANIFEST, FROZEN_MANIFEST)
    manifest_hash = hashlib.sha256(FROZEN_MANIFEST.read_bytes()).hexdigest()
    (FROZEN_MANIFEST.with_suffix(".sha256")).write_text(
        f"{manifest_hash}  {FROZEN_MANIFEST.name}\n", encoding="utf-8"
    )

    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    heldout = load_heldout_run_ids(FROZEN_MANIFEST)
    all_epochs = sorted(
        path for path in EPOCH_DIR.glob("*-epo.fif")
        if not path.name.startswith("._")
    )
    current_sg_values = [path for path in all_epochs if "_Train" in path.name]
    current_overlap = sorted(
        run_id_from_epoch_path(path) for path in current_sg_values
        if run_id_from_epoch_path(path) in heldout
    )
    current_test_labeled = sorted(
        run_id_from_epoch_path(path) for path in current_sg_values
        if "_Test" in run_id_from_epoch_path(path)
    )
    clean_train = [
        path for path in current_sg_values
        if run_id_from_epoch_path(path) not in heldout
    ]
    assert_training_pool_disjoint(clean_train, heldout)

    target_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"target": 0, "non_target": 0, "runs": 0}
    )
    for path in clean_train:
        subject = path.name.split("_SE", 1)[0]
        epochs = mne.read_epochs(path, preload=False, verbose="ERROR")
        labels = epochs.events[:, 2]
        target_counts[subject]["target"] += int((labels == 1).sum())
        target_counts[subject]["non_target"] += int((labels == 0).sum())
        target_counts[subject]["runs"] += 1
        del epochs

    training_conditions = Counter(condition(run_id_from_epoch_path(p)) for p in clean_train)
    manifest_conditions = Counter(condition(run_id) for run_id in heldout)
    prior_conditions = Counter(condition(run_id_from_epoch_path(p)) for p in current_sg_values)
    cnn_metadata = json.loads(CNN_METADATA.read_text(encoding="utf-8"))
    cnn_training_runs = set(cnn_metadata.get("train_sessions", []))
    report = {
        "manifest_path": str(FROZEN_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "heldout_runs": len(heldout),
        "heldout_subjects": len(manifest),
        "processed_runs": len(all_epochs),
        "existing_swlda_training_glob": "D_*_SE001*Train*-epo.fif",
        "existing_swlda_training_runs": len(current_sg_values),
        "existing_swlda_manifest_overlap_runs": current_overlap,
        "existing_swlda_test_labelled_runs": current_test_labeled,
        "existing_cnn_training_run_count": len(cnn_training_runs),
        "existing_cnn_manifest_overlap_runs": sorted(cnn_training_runs & heldout),
        "existing_cnn_test_labelled_run_count": sum(
            "_Test" in run_id for run_id in cnn_training_runs
        ),
        "existing_cnn_note": (
            "Existing CNN metadata lists Test-labelled training runs; retrain on "
            "the clean RC/Train pool before using it as a classifier control."
        ),
        "clean_training_pool_runs": len(clean_train),
        "current_swlda_pool_condition_counts": dict(sorted(prior_conditions.items())),
        "clean_training_condition_counts": dict(sorted(training_conditions.items())),
        "manifest_condition_counts": dict(sorted(manifest_conditions.items())),
        "condition_shift_note": (
            "Clean training pool is RC/Train only; held-out manifest contains "
            "Dyn and DynBigram runs plus 19 RC/Train runs."
        ),
        "clean_training_epoch_counts_by_subject": dict(sorted(target_counts.items())),
        "training_pool_run_ids": sorted(run_id_from_epoch_path(p) for p in clean_train),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {
        "clean_training_epoch_counts_by_subject", "training_pool_run_ids"
    }}, indent=2))
    print(f"Audit written to {RESULT_PATH}")


if __name__ == "__main__":
    main()
