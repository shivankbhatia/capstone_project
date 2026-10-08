#!/usr/bin/env python3
"""Run the metadata and target-integrity audit for the frozen Q split.

This is an evaluation-side audit. It may read the label vault to verify
ground-truth extraction, but it never fits a classifier or computes model
scores. Classifier/training sources must not import this module or the vault.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
VAULT = ROOT / "data/evaluation/ground_truth_vault.json"
TRAIN_REGISTRY = ROOT / "data/processed/study_q_train_registry.json"
MANIFEST = ROOT / "splits/study_q_manifest.json"
OUT = ROOT / "results/tables/study_q_audit.json"
CONDITIONS = ("ColorIntensification", "Grey-to-Color", "Grey-to-White")


def _word_count(text: str) -> int:
    """Count whitespace-separated words; Q key labels may limit interpretation."""
    return len(re.findall(r"\S+", text.strip()))


def audit() -> dict:
    import mne
    from src.data.batch_preprocess import parse_bigp3bci_edf

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    heldout = {run for runs in manifest.values() for run in runs}
    vault = json.loads(VAULT.read_text(encoding="utf-8"))
    training_registry = json.loads(TRAIN_REGISTRY.read_text(encoding="utf-8"))
    paths = sorted(p for p in DATA.rglob("*.edf")
                   if p.stem in heldout and not p.name.startswith("._"))
    if {p.stem for p in paths} != heldout:
        raise RuntimeError("Frozen Q manifest does not resolve to the expected EDF files")
    if not heldout <= vault.keys():
        raise RuntimeError("A held-out Q run is missing its evaluation-vault label")

    subject_sessions: dict[str, set[str]] = defaultdict(set)
    subject_conditions: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sequence_counts: dict[str, list[int]] = defaultdict(list)
    condition_sequence_counts: dict[str, list[int]] = defaultdict(list)
    chars_by_subject: Counter[str] = Counter()
    words_by_subject: Counter[str] = Counter()
    extraction_matches = 0
    extraction_mismatches = []
    run_records = []
    geometry = Counter()
    errors = []

    # Session, split, condition, and run counts are collected from paths only.
    source_counts: dict[str, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    for source_path in DATA.rglob("*.edf"):
        if source_path.name.startswith("._"):
            continue
        source_subject, source_session, source_split, source_condition, _ = (
            source_path.relative_to(DATA).parts
        )
        source_counts[source_subject][source_session][source_split][source_condition] += 1

    for index, path in enumerate(paths, start=1):
        subject, session, _, condition, _ = path.relative_to(DATA).parts
        subject_sessions[subject].add(session)
        subject_conditions[subject][session].add(condition)
        target = str(vault[path.stem])
        chars_by_subject[subject] += len(target)
        words_by_subject[subject] += _word_count(target)
        try:
            # Capture parser status output; only aggregate it into this report.
            with contextlib.redirect_stdout(io.StringIO()):
                epochs, extracted, info = parse_bigp3bci_edf(path, verbose=False)
            normalized_extracted = str(extracted).replace(" ", "").upper()
            normalized_target = target.replace(" ", "").upper()
            matches = normalized_extracted == normalized_target
            extraction_matches += int(matches)
            if not matches:
                extraction_mismatches.append(path.stem)
            md = epochs.metadata
            if md is None or not {"char_index", "sequence_in_char"}.issubset(md.columns):
                counts = []
            else:
                counts = md.groupby("char_index")["sequence_in_char"].nunique().astype(int).tolist()
            valid_counts = [int(x) for x in counts if x > 0]
            sequence_counts[path.stem] = valid_counts
            condition_sequence_counts[condition].extend(valid_counts)
            geometry[(tuple(epochs.ch_names), round(float(epochs.info["sfreq"]), 2),
                      int(info["n_rows"]), int(info["n_cols"]),
                      bool(info["group_flash"]), int(info["flashes_per_seq"]))] += 1
            run_records.append({
                "run_id": path.stem,
                "subject": subject,
                "session": session,
                "condition": condition,
                "selected_characters": int(info["agreement"].get("n_characters_detected", 0)),
                "target_symbols_in_vault": len(target),
                "sequence_counts_per_character": valid_counts,
                "sequence_count_min": min(valid_counts) if valid_counts else None,
                "sequence_count_max": max(valid_counts) if valid_counts else None,
                "extraction_matches_vault": matches,
            })
            del epochs
        except Exception as exc:  # report all bad runs without losing the audit
            errors.append({"run_id": path.stem, "error": f"{type(exc).__name__}: {exc}"})
        if index % 20 == 0 or index == len(paths):
            print(f"Audited {index}/{len(paths)} frozen Test runs", flush=True)

    sessions_hist = Counter(len(value) for value in subject_sessions.values())
    geometry_out = []
    for (channels, sfreq, rows, cols, group_flash, flashes), count in geometry.items():
        geometry_out.append({
            "runs": count, "channel_count": len(channels), "channel_names": list(channels),
            "sfreq_hz": sfreq, "grid": [rows, cols], "group_flash": group_flash,
            "flashes_per_sequence": flashes,
        })

    def distribution(values: list[int]) -> dict:
        if not values:
            return {"count": 0}
        return {
            "count": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
        }

    by_condition = {
        condition: distribution(condition_sequence_counts[condition])
        for condition in CONDITIONS
    }
    per_subject = {
        subject: {
            "session_count_in_dataset": len(subject_sessions[subject]),
            "test_runs_in_manifest": sum(1 for r in run_records if r["subject"] == subject),
            "vault_text_entries": sum(1 for p in paths if p.stem.startswith(subject + "_")),
            "whitespace_delimited_words_in_vault": int(words_by_subject[subject]),
            "vault_symbol_characters": int(chars_by_subject[subject]),
            "clean_train_pool_text_entries": sum(
                1 for run_id in training_registry if run_id.startswith(subject + "_")
            ),
            "clean_train_pool_whitespace_delimited_words": sum(
                _word_count(str(text)) for run_id, text in training_registry.items()
                if run_id.startswith(subject + "_")
            ),
            "clean_train_pool_symbol_characters": sum(
                len(str(text)) for run_id, text in training_registry.items()
                if run_id.startswith(subject + "_")
            ),
        }
        for subject in sorted(subject_sessions)
    }
    run_condition_counts = {
        subject: {
            session: {
                split: dict(sorted(condition_counts.items()))
                for split, condition_counts in sorted(by_split.items())
            }
            for session, by_split in sorted(by_session.items())
        }
        for subject, by_session in sorted(source_counts.items())
    }
    all_depths = [depth for counts in condition_sequence_counts.values() for depth in counts]
    result = {
        "study": "StudyQ",
        "split": "SE003 Test only; same-session Train runs excluded; no classifier scores computed",
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_run_count": len(paths),
        "session_count_distribution": {str(k): v for k, v in sorted(sessions_hist.items())},
        "session_condition_counts": {
            subject: {session: sorted(values) for session, values in sorted(by_session.items())}
            for subject, by_session in sorted(subject_conditions.items())
        },
        "source_run_counts_by_subject_session_split_condition": run_condition_counts,
        "condition_totals_selected_for_evaluation": dict(sorted(Counter(r["condition"] for r in run_records).items())),
        "recording_geometry": geometry_out,
        "sequence_counts_per_character_by_condition": by_condition,
        "stopping_depth_coverage": {
            "locked_max_sequences": 10,
            "characters_with_fewer_than_max_sequences_recorded": sum(x < 10 for x in all_depths),
            "characters_with_max_sequences_recorded": sum(x == 10 for x in all_depths),
            "depth_histogram": {str(k): v for k, v in sorted(Counter(all_depths).items())},
            "right_censored_for_max_10_replay": sum(x < 10 for x in all_depths) > 0,
        },
        "ground_truth_extraction": {
            "runs_compared_with_vault": extraction_matches + len(extraction_mismatches),
            "exact_matches": extraction_matches,
            "mismatches": extraction_mismatches,
            "errors": errors,
        },
        "text_volume": {
            "whitespace_tokenization_caveat": "Q ground-truth strings encode keyboard key labels; whitespace word counts may not represent natural-language words.",
            "clean_train_pool_definition": "Train labels outside held-out session SE003; sourced only from the dedicated Q training registry.",
            "per_subject": per_subject,
        },
        "run_records": run_records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps({k: v for k, v in result.items() if k != "run_records"}, indent=2))
