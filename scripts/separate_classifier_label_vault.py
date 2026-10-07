#!/usr/bin/env python3
"""Move evaluation targets out of the training-visible registry.

Only clean Study D RC/Train labels outside the frozen manifest remain in
``ground_truth_registry.json``. All other session targets are moved to the
evaluation-only vault. This script reports counts only and never prints text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_REGISTRY = ROOT / "data/processed/ground_truth_registry.json"
VAULT = ROOT / "data/evaluation/ground_truth_vault.json"
MANIFEST = ROOT / "splits/heldout_manifest.json"


def partition_registry(current: dict, vault: dict, heldout_ids: set[str]):
    """Return clean Study D calibration labels and the evaluation-only labels."""
    visible, protected = {}, dict(vault)
    for run_id, target in current.items():
        safe_train = (
            str(run_id).startswith("D_")
            and "_RC_Train" in str(run_id)
            and "_Test" not in str(run_id)
            and run_id not in heldout_ids
        )
        if safe_train:
            visible[run_id] = target
            protected.pop(run_id, None)
        else:
            protected[run_id] = target
    return visible, protected


def split_registry(apply: bool = False) -> dict:
    with TRAIN_REGISTRY.open(encoding="utf-8") as handle:
        current = json.load(handle)
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    heldout_ids = {run for runs in manifest.values() for run in runs}

    if VAULT.exists():
        with VAULT.open(encoding="utf-8") as handle:
            vault = json.load(handle)
    else:
        vault = {}

    # Keep only Study D, non-heldout, RC/Train sessions. Seal all Study D
    # Test sessions, frozen runs (including RC/Train members), and all other
    # studies, which are reserved for external validation.
    visible, protected = partition_registry(current, vault, heldout_ids)

    if apply:
        VAULT.parent.mkdir(parents=True, exist_ok=True)
        VAULT.write_text(json.dumps(protected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        TRAIN_REGISTRY.write_text(json.dumps(visible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        VAULT.chmod(0o600)

    return {
        "training_visible_sessions": len(visible),
        "evaluation_vault_sessions": len(protected),
        "frozen_manifest_sessions": len(heldout_ids),
        "applied": apply,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write the split; otherwise report counts only")
    args = parser.parse_args()
    print(json.dumps(split_registry(apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
