#!/usr/bin/env python3
"""One-shot, locked as-recorded-depth evaluation on Study Q SE003 Test.

This evaluator is intentionally the only classifier evaluation script allowed
to open the Q label vault. It requires an explicit flag, verifies the frozen
manifest and lock, and creates an exclusive run marker before reading labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
from scipy.special import softmax
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_classical_cv import _fit_m0_streaming, _load_transform_batch  # noqa: E402
from scripts.run_study_q_classifier_cv import M0_PREPROCESSING, _decode  # noqa: E402
from src.data.batch_preprocess import parse_bigp3bci_edf  # noqa: E402
from src.evaluation.sequence_scoring import aggregate_membership_logits  # noqa: E402

LOCK = ROOT / "splits/studyq_classifier_lock.json"
MANIFEST = ROOT / "splits/study_q_manifest.json"
VAULT = ROOT / "data/evaluation/ground_truth_vault.json"
CACHE = ROOT / "data/cache/study_q_classifier_epochs.h5"
EDF_ROOT = ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
OUTPUT = ROOT / "results/tables/study_q_test_once.json"
MARKER = OUTPUT.with_suffix(".started")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    count = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def _bootstrap(values: np.ndarray, seed: int = 1907, reps: int = 10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n_subjects": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "n_subjects": int(len(values))}


def _paired_test(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)


def _char_records(logits, meta, targets, grid, condition, alpha, temperature):
    records = []
    for char_idx in sorted(set(meta["char_idx"].tolist())):
        if char_idx < 0:
            continue
        indices = np.flatnonzero(meta["char_idx"] == char_idx)
        if not len(indices):
            continue
        seqs = sorted(s for s in set(meta["seq_idx"][indices].tolist()) if s >= 0)
        if not seqs:
            continue
        # The target character is independently supplied by the sealed vault.
        # A mismatch is a data-integrity failure, not an opportunity to repair.
        if char_idx >= len(targets):
            raise ValueError(f"Vault target is shorter than parsed character index {char_idx}")
        symbol = targets[char_idx]
        match = np.flatnonzero(grid.reshape(-1) == symbol)
        if len(match) != 1:
            # Grid label comparison is case-insensitive for letter keys only.
            match = np.flatnonzero(np.char.upper(grid.astype(str)).reshape(-1) == str(symbol).upper())
        if len(match) != 1:
            raise ValueError(f"Vault character {symbol!r} does not map uniquely to Study Q grid")
        target = int(match[0])

        # Teacher-forced context matches the clean inner-OOF fusion convention.
        from src.models.llm_predictor import LLMPredictor
        prior_model = _char_records.prior_model
        context = ""
        for prior_symbol in targets[:char_idx]:
            context += prior_model.special_key_map.get(prior_symbol, str(prior_symbol).lower())
        if context not in _char_records.prior_cache:
            _char_records.prior_cache[context] = prior_model.predict_next_char(context)
        prior = _char_records.prior_cache[context]
        prior = np.asarray(prior, dtype=float).reshape(-1)
        if len(prior) != 72:
            raise ValueError(f"Expected a 72-cell prior, received {len(prior)}")
        log_prior = np.full(72, -1e9, dtype=float)
        positive = prior > 0
        log_prior[positive] = alpha * np.log(np.clip(prior[positive], 1e-12, 1.0))

        cumulative = np.zeros(72, dtype=float)
        fused_cumulative = log_prior.copy()
        by_depth = {}
        for depth, seq in enumerate(seqs, start=1):
            ix = indices[meta["seq_idx"][indices] == seq]
            if not len(ix):
                continue
            sequence_post = aggregate_membership_logits(
                logits[ix] / temperature, meta["lit_mask"][ix], n_rows=9, n_cols=8,
            )
            log_evidence = np.log(np.clip(sequence_post, 1e-12, 1.0))
            cumulative += log_evidence
            fused_cumulative += log_evidence
            by_depth[depth] = {
                "eeg_guess": int(softmax(cumulative).argmax()),
                "fused_guess": int(softmax(fused_cumulative).argmax()),
            }
        records.append({"subject": meta["subject"], "run_id": meta["run_id"],
                        "condition": condition, "char_idx": int(char_idx),
                        "depth": len(seqs), "target_cell": target,
                        "eeg_correct": int(int(softmax(cumulative).argmax()) == target),
                        "fused_correct": int(int(softmax(fused_cumulative).argmax()) == target),
                        "by_depth": by_depth})
    return records


def _summarize(records):
    groups = {"overall": records}
    for condition in sorted({row["condition"] for row in records}):
        groups[condition] = [row for row in records if row["condition"] == condition]
    metrics, pvalues = {}, {}
    for group, rows in groups.items():
        subject_rows = {}
        for row in rows:
            per = subject_rows.setdefault(row["subject"], {"eeg": [], "fused": [], "delta": []})
            per["eeg"].append(row["eeg_correct"])
            per["fused"].append(row["fused_correct"])
        per_subject = {s: {"n_characters": len(v["eeg"]),
                           "eeg_accuracy": float(np.mean(v["eeg"])),
                           "fused_accuracy": float(np.mean(v["fused"])),
                           "paired_delta": float(np.mean(v["fused"]) - np.mean(v["eeg"]))}
                       for s, v in subject_rows.items()}
        deltas = np.asarray([v["paired_delta"] for v in per_subject.values()])
        pvalues[group] = _paired_test(deltas)
        metrics[group] = {
            "characters": len(rows), "subjects": len(per_subject),
            "eeg_accuracy": float(np.mean([r["eeg_correct"] for r in rows])) if rows else None,
            "fused_accuracy": float(np.mean([r["fused_correct"] for r in rows])) if rows else None,
            "paired_fused_minus_eeg_subject_bootstrap": _bootstrap(deltas),
            "wilcoxon_p": pvalues[group], "per_subject": per_subject,
        }
        depths = {}
        for row in rows:
            depths[str(row["depth"])] = depths.get(str(row["depth"]), 0) + 1
        metrics[group]["depth_distribution_characters"] = dict(sorted(depths.items(), key=lambda x: int(x[0])))
    adjusted = _holm(pvalues)
    for group in metrics:
        metrics[group]["wilcoxon_holm_p"] = adjusted[group]

    buckets = {}
    for name, allowed in {"4-6": {4, 5, 6}, "7-8": {7, 8}, "9-10": {9, 10}}.items():
        rows = [r for r in records if r["depth"] in allowed]
        by_subject = {}
        for row in rows:
            per = by_subject.setdefault(row["subject"], {"eeg": [], "fused": []})
            per["eeg"].append(row["eeg_correct"])
            per["fused"].append(row["fused_correct"])
        deltas = np.asarray([np.mean(v["fused"]) - np.mean(v["eeg"]) for v in by_subject.values()])
        buckets[name] = {"characters": len(rows), "paired_delta": _bootstrap(deltas),
                         "positive_subjects": int(np.sum(deltas > 0)),
                         "nonnegative_mean": bool(np.mean(deltas) >= 0) if len(deltas) else None}
    curve = {}
    for depth in range(1, 11):
        rows = [r for r in records if r["depth"] >= depth]
        eeg, fused = [], []
        for row in rows:
            values = row["by_depth"].get(depth)
            if values:
                eeg.append(values["eeg_guess"] == row["target_cell"])
                fused.append(values["fused_guess"] == row["target_cell"])
        curve[str(depth)] = {"characters_with_prefix": len(rows),
                             "eeg_accuracy": float(np.mean(eeg)) if eeg else None,
                             "fused_accuracy": float(np.mean(fused)) if fused else None,
                             "label": "descriptive; biased by early stopping and unequal recorded depth"}
    return {"by_condition_and_overall": metrics,
            "depth_bucket_sensitivity": buckets,
            "accuracy_vs_depth_biased_descriptive": curve}


def run(*, allow_heldout_eval: bool, dry_run_train: bool = False):
    if not allow_heldout_eval and not dry_run_train:
        raise PermissionError("Pass --allow-heldout-eval only for the preregistered one-shot SE003 evaluation")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    manifest_bytes = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != lock["manifest_sha256"]:
        raise RuntimeError("Manifest hash differs from the preregistered lock")
    manifest = json.loads(manifest_bytes)
    heldout_ids = {run_id for ids in manifest.values() for run_id in ids}
    if dry_run_train:
        return {"dry_run": True, "manifest_verified": True, "test_labels_read": False}
    if OUTPUT.exists() or MARKER.exists():
        raise FileExistsError("One-shot guard: output or run marker already exists")
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(),
                                 "manifest_sha256": lock["manifest_sha256"]}) + "\n")

    # Labels are opened only after the explicit one-shot call and exclusive marker.
    vault = json.loads(VAULT.read_text(encoding="utf-8"))
    targets = {key: value for key, value in vault.items() if key in heldout_ids}
    if set(targets) != heldout_ids:
        raise ValueError("Vault labels do not exactly cover the frozen manifest")
    files = {path.stem: path for path in EDF_ROOT.rglob("*.edf") if path.stem in heldout_ids}
    if set(files) != heldout_ids:
        raise ValueError("EDF paths do not exactly cover the frozen manifest")

    with h5py.File(CACHE, "r") as h5:
        y = h5["y"][:].astype(np.uint8)
        run_ids = _decode(h5["run_id"][:])
        all_idx = np.arange(len(y))
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        cfg = dict(M0_PREPROCESSING)
        cfg["decim"] = 1  # cache and Test epochs are already decimated by two
        model, scaler, _, _, n_train = _fit_m0_streaming(
            h5["X"], all_idx, all_idx[:1], y, cfg, channels, sfreq, tmin, seed=42,
        )

    layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text(encoding="utf-8"))["StudyQ"]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for symbol, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = symbol
    from src.models.llm_predictor import LLMPredictor
    _char_records.prior_model = LLMPredictor(grid, local_files_only=True)
    _char_records.prior_cache = {}

    temperature = float(lock["calibration"]["temperature"])
    alpha = float(lock["fusion"]["rung_1"]["alpha"])
    records = []
    for i, run_id in enumerate(sorted(heldout_ids), start=1):
        path = files[run_id]
        epochs, parsed_targets, info = parse_bigp3bci_edf(
            path, tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
            verbose=False, baseline_correction=False, reject_threshold=None, decim=2,
        )
        if info["n_rows"] != 9 or info["n_cols"] != 8 or not info["group_flash"]:
            raise ValueError(f"Unexpected Study Q grid/scoring for {run_id}")
        if set(channels) != set(epochs.ch_names):
            raise ValueError(f"EEG channel set differs from the locked cache for {run_id}")
        epochs.reorder_channels(channels)
        if not np.isclose(float(epochs.info["sfreq"]), sfreq):
            raise ValueError(f"Sampling rate differs from the locked cache for {run_id}")
        X = epochs.get_data(copy=True)
        meta = epochs.metadata
        masks = np.asarray([[int(bit) for bit in mask] for mask in meta["lit_mask"].astype(str)])
        features, keep = _load_transform_batch(X, cfg, channels, sfreq, tmin)
        scores = np.full(len(X), np.nan)
        scores[keep] = model.decision_function(scaler.transform(features[keep]))
        valid = keep & (meta["char_index"].to_numpy() >= 0) & (meta["sequence_in_char"].to_numpy() >= 0)
        test_meta = {"subject": run_id.split("_SE", 1)[0], "run_id": run_id,
                     "char_idx": meta["char_index"].to_numpy()[valid].astype(int),
                     "seq_idx": meta["sequence_in_char"].to_numpy()[valid].astype(int),
                     "lit_mask": masks[valid]}
        targets_for_run = targets[run_id]
        # Verify the parsed target stream and sealed target sequence agree.
        parsed = "".join(str(s) for s in targets_for_run)
        if parsed_targets.upper() != parsed.upper():
            raise ValueError(f"Parsed EDF target string disagrees with the sealed vault for {run_id}")
        if len(set(test_meta["char_idx"])) > len(parsed):
            raise ValueError(f"Parsed character indices exceed vault target length for {run_id}")
        condition = path.parent.name
        records.extend(_char_records(scores[valid], test_meta, parsed, grid, condition,
                                     alpha, temperature))
        if i % 10 == 0:
            print(f"Q SE003 one-shot: parsed {i}/{len(heldout_ids)} runs", flush=True)

    if not records:
        raise ValueError("No valid Test characters were scored")
    result = {
        "study": "StudyQ", "evaluation": "one-shot frozen SE003 Test; as-recorded depth",
        "test_data_used_for_tuning": False, "test_labels_read": True,
        "manifest_sha256": lock["manifest_sha256"], "lock_sha256": sha256(LOCK),
        "training_cache_sha256": sha256(CACHE), "git_hash": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "temperature": temperature, "fusion_alpha": alpha,
        "training_epochs": int(n_train), "test_runs": len(heldout_ids),
        "n_characters_scored": len(records), "metrics": _summarize(records),
        "character_records": records,
        "prohibited_claims": ["stopping policy", "flashes per character", "ITR"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-heldout-eval", action="store_true")
    parser.add_argument("--dry-run-train", action="store_true",
                        help="verify lock and manifest without opening Test labels or EEG")
    args = parser.parse_args()
    value = run(allow_heldout_eval=args.allow_heldout_eval, dry_run_train=args.dry_run_train)
    print(json.dumps({k: v for k, v in value.items() if k not in {"character_records", "metrics"}}, indent=2))
