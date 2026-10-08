#!/usr/bin/env python3
"""Inventory Study Q and freeze its pre-existing Train/Test run split.

This audit uses paths only. It never opens EDF files or reads their trigger,
target, or EEG channels. Test runs are represented in a manifest by run ID so
later training guards can exclude them before any model is fit.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
MANIFEST = ROOT / "splits/study_q_manifest.json"
EXCLUDED_TRAIN = ROOT / "splits/study_q_excluded_training_runs.json"
REPORT = ROOT / "results/tables/study_q_path_audit.json"
HELDOUT_SESSION = "SE003"


def audit(*, refreeze: bool = False) -> dict:
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    files = sorted(p for p in DATA.rglob("*.edf") if not p.name.startswith("._"))
    manifest: dict[str, list[str]] = defaultdict(list)
    excluded_train: dict[str, list[str]] = defaultdict(list)
    counts = Counter()
    subjects: set[str] = set()
    for path in files:
        # StudyQ/<subject>/<session>/<split>/<condition>/<run>.edf
        if len(path.relative_to(DATA).parts) != 5:
            raise ValueError(f"Unexpected Study Q path layout: {path}")
        subject, session, split, condition, _ = path.relative_to(DATA).parts
        if split not in {"Train", "Test"}:
            raise ValueError(f"Unexpected split name {split!r}: {path}")
        run_id = path.stem
        subjects.add(subject)
        counts[(split, condition)] += 1
        if split == "Test":
            if session == HELDOUT_SESSION:
                manifest[subject].append(run_id)
        elif split == "Train" and session == HELDOUT_SESSION:
            excluded_train[subject].append(run_id)

    normalized = {key: sorted(values) for key, values in sorted(manifest.items())}
    excluded = {key: sorted(values) for key, values in sorted(excluded_train.items())}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and not refreeze:
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError("Frozen Study Q manifest differs from current path inventory")
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    (MANIFEST.with_suffix(".sha256")).write_text(f"{digest}  {MANIFEST.name}\n", encoding="utf-8")
    if EXCLUDED_TRAIN.exists() and not refreeze:
        existing_excluded = json.loads(EXCLUDED_TRAIN.read_text(encoding="utf-8"))
        if existing_excluded != excluded:
            raise RuntimeError("Frozen Study Q session exclusions differ from current path inventory")
    else:
        EXCLUDED_TRAIN.write_text(json.dumps(excluded, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    excluded_digest = hashlib.sha256(EXCLUDED_TRAIN.read_bytes()).hexdigest()
    report = {
        "study": "StudyQ",
        "audit_scope": "EDF paths and names only; no file contents opened",
        "raw_edf_count": len(files),
        "subject_count": len(subjects),
        "manifest_run_count": sum(map(len, normalized.values())),
        "official_test_run_count": sum(n for (split, _), n in counts.items() if split == "Test"),
        "training_run_count": sum(n for (split, _), n in counts.items() if split == "Train"),
        "evaluation_session": HELDOUT_SESSION,
        "subjects_with_evaluation_session": len(normalized),
        "excluded_same_session_train_run_count": sum(map(len, excluded.values())),
        "session_counts_per_subject": _session_count_distribution(files, DATA),
        "split_condition_counts": {
            f"{split}/{condition}": n
            for (split, condition), n in sorted(counts.items())
        },
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": digest,
        "excluded_training_runs": str(EXCLUDED_TRAIN.relative_to(ROOT)),
        "excluded_training_runs_sha256": excluded_digest,
        "run_leakage_unit": "session (SE003 held out; same-session Train runs excluded)",
        "warning": "This is a frozen path-derived split only; no Q EEG or labels have been evaluated.",
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _session_count_distribution(files: list[Path], root: Path) -> dict[str, int]:
    """Count distinct session directories for each subject from paths only."""
    sessions: dict[str, set[str]] = defaultdict(set)
    for path in files:
        subject, session = path.relative_to(root).parts[:2]
        sessions[subject].add(session)
    return {str(n): count for n, count in sorted(Counter(map(len, sessions.values())).items())}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refreeze", action="store_true", help="Replace the provisional path-derived split before any Q evaluation")
    print(json.dumps(audit(refreeze=parser.parse_args().refreeze), indent=2))
