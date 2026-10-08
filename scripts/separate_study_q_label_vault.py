#!/usr/bin/env python3
"""Create a Q-only training registry and keep Test/session labels vaulted.

The global training registry remains Study-D-only. Q training labels are
written to a dedicated registry containing only Train runs outside SE003;
all other Q labels remain in the evaluation vault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
VAULT = ROOT / "data/evaluation/ground_truth_vault.json"
TRAIN_REGISTRY = ROOT / "data/processed/study_q_train_registry.json"
GLOBAL_REGISTRY = ROOT / "data/processed/ground_truth_registry.json"
MANIFEST = ROOT / "splits/study_q_manifest.json"
EXCLUDED = ROOT / "splits/study_q_excluded_training_runs.json"
REPORT = ROOT / "results/tables/study_q_vault_separation.json"


def separate(*, apply: bool = False) -> dict:
    vault = json.loads(VAULT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    heldout = {run for runs in manifest.values() for run in runs}
    exclusions = json.loads(EXCLUDED.read_text(encoding="utf-8"))
    excluded_train = {run for runs in exclusions.values() for run in runs}
    clean_train = {
        p.stem for p in DATA.rglob("*.edf")
        if "/Train/" in p.as_posix() and "_SE003_" not in p.stem
        and not p.name.startswith("._")
    }
    if clean_train & (heldout | excluded_train):
        raise ValueError("Q clean training runs overlap a frozen evaluation/exclusion set")
    if any("_Test" in run for run in clean_train):
        raise ValueError("Q training registry candidate includes a Test-labelled run")
    if not heldout <= vault.keys():
        raise ValueError("A frozen Study Q Test label is missing from the evaluation vault")
    if GLOBAL_REGISTRY.exists():
        global_registry = json.loads(GLOBAL_REGISTRY.read_text(encoding="utf-8"))
        if any(run.startswith("Q_") for run in global_registry):
            raise ValueError("The global registry contains Q labels; resolve overlap before Q separation")

    existing = json.loads(TRAIN_REGISTRY.read_text(encoding="utf-8")) if TRAIN_REGISTRY.exists() else {}
    missing = clean_train - vault.keys()
    if missing - existing.keys():
        raise ValueError(f"Q training labels unavailable in vault/registry for {len(missing-existing)} runs")
    train_labels = {run: vault[run] for run in clean_train if run in vault}
    train_labels.update({run: existing[run] for run in clean_train if run in existing})
    if set(train_labels) != clean_train:
        raise ValueError("Could not construct the exact clean Q Train label set")
    if set(train_labels) & (heldout | excluded_train):
        raise ValueError("Q training labels overlap held-out labels or same-session exclusions")
    next_vault = {run: target for run, target in vault.items() if run not in clean_train}
    if heldout - next_vault.keys():
        raise ValueError("Refusing to remove a Q Test label from the vault")

    if apply:
        TRAIN_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_REGISTRY.write_text(json.dumps(train_labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        TRAIN_REGISTRY.chmod(0o600)
        VAULT.write_text(json.dumps(next_vault, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        VAULT.chmod(0o600)
    report = {
        "study": "StudyQ",
        "applied": apply,
        "training_registry": str(TRAIN_REGISTRY.relative_to(ROOT)),
        "clean_train_run_count": len(train_labels),
        "clean_train_test_overlap": 0,
        "clean_train_same_session_exclusion_overlap": 0,
        "frozen_test_labels_remaining_in_vault": len(heldout & next_vault.keys()),
        "vault_q_labels_before": sum(run.startswith("Q_") for run in vault),
        "vault_q_labels_after": sum(run.startswith("Q_") for run in next_vault),
        "global_registry_contains_q_labels": False,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "training_registry_sha256": hashlib.sha256((json.dumps(train_labels, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
        "vault_sha256_after": hashlib.sha256((json.dumps(next_vault, indent=2, sort_keys=True) + "\n").encode()).hexdigest(),
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the isolated training registry and remove those labels from the vault")
    print(json.dumps(separate(apply=parser.parse_args().apply), indent=2))
