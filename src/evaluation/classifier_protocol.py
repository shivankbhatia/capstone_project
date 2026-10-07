"""Leakage guards shared by classifier experiments."""

from __future__ import annotations

from pathlib import Path
import json


def load_heldout_run_ids(manifest_path: str | Path) -> set[str]:
    """Load the frozen set of run IDs from the subject-keyed manifest."""
    with Path(manifest_path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return {str(run_id) for run_ids in manifest.values() for run_id in run_ids}


def run_id_from_epoch_path(path: str | Path) -> str:
    """Return the manifest-compatible run ID for an epoched FIF path."""
    return Path(path).name.removesuffix("-epo.fif")


def assert_training_pool_disjoint(
    training_paths: list[str | Path], heldout_run_ids: set[str]
) -> None:
    """Fail closed if a training file is held out or is labelled as Test."""
    violations = []
    for path in training_paths:
        run_id = run_id_from_epoch_path(path)
        if run_id in heldout_run_ids:
            violations.append(f"held-out run: {run_id}")
        if "_Test" in run_id:
            violations.append(f"Test-labelled run: {run_id}")
    if violations:
        raise ValueError("Invalid classifier training pool: " + "; ".join(violations))


def clean_training_paths(
    paths: list[str | Path], heldout_run_ids: set[str]
) -> list[Path]:
    """Select calibration/training runs outside the frozen manifest."""
    selected = [Path(path) for path in paths]
    assert_training_pool_disjoint(selected, heldout_run_ids)
    return selected
