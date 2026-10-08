#!/usr/bin/env python3
"""Nested session-grouped Q M0 calibration and fixed-prior fusion OOF.

Each outer validation session is excluded from both classifier and calibration
fits. Inner OOF logits from the outer training sessions fit one temperature
using sequence-level grid NLL. The fixed D fusion weight/stopping policy are
then applied to the outer-session OOF scores. No Q Test EDFs or vault labels
are read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from src.evaluation.classifier_protocol import assert_training_pool_disjoint  # noqa: E402
from src.evaluation.sequence_scoring import aggregate_membership_logits  # noqa: E402

from scripts.run_study_q_classifier_cv import M0_PREPROCESSING, _decode  # noqa: E402

MANIFEST = ROOT / "splits/study_q_manifest.json"
EXCLUSIONS = ROOT / "splits/study_q_excluded_training_runs.json"


def _groups(meta):
    return np.char.add(np.char.add(meta["subject"], "_"), meta["session"])


def _true_cells(meta, rows: np.ndarray) -> dict[tuple[str, int], int]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index in rows:
        key = (str(meta["run_id"][index]), int(meta["char_idx"][index]))
        if key[1] >= 0:
            groups.setdefault(key, []).append(int(index))
    result = {}
    for key, indices in groups.items():
        indices = np.asarray(indices, dtype=int)
        target_masks = meta["lit_mask"][indices[meta["y"][indices] == 1]].astype(bool)
        if not len(target_masks):
            continue
        intersection = np.all(target_masks, axis=0)
        if int(intersection.sum()) == 1:
            result[key] = int(np.flatnonzero(intersection)[0])
    return result


def _sequence_nll(log_temperature, logits, meta, rows):
    temperature = float(np.exp(log_temperature))
    true_cells = _true_cells(meta, rows)
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index in rows:
        key = (str(meta["run_id"][index]), int(meta["char_idx"][index]), int(meta["seq_idx"][index]))
        if key[1] >= 0 and key[2] >= 0 and key[:2] in true_cells:
            grouped.setdefault(key, []).append(int(index))
    losses = []
    for (run_id, char_idx, _), indices in grouped.items():
        ix = np.asarray(indices, dtype=int)
        posterior = aggregate_membership_logits(
            logits[ix] / temperature, meta["lit_mask"][ix], n_rows=9, n_cols=8,
        )
        losses.append(-np.log(np.clip(posterior[true_cells[(run_id, char_idx)]], 1e-12, 1.0)))
    return float(np.mean(losses)) if losses else float("inf")


def _sequence_metrics(logits, meta, rows, n_bins=10):
    """Top-label ECE and multiclass Brier score at the decoder input level."""
    true_cells = _true_cells(meta, rows)
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for index in rows:
        key = (str(meta["run_id"][index]), int(meta["char_idx"][index]), int(meta["seq_idx"][index]))
        if key[1] >= 0 and key[2] >= 0 and key[:2] in true_cells:
            grouped.setdefault(key, []).append(int(index))
    confidences, correct, briers, nlls = [], [], [], []
    for (run_id, char_idx, _), indices in grouped.items():
        ix = np.asarray(indices, dtype=int)
        posterior = aggregate_membership_logits(
            logits[ix], meta["lit_mask"][ix], n_rows=9, n_cols=8,
        )
        target = true_cells[(run_id, char_idx)]
        one_hot = np.zeros(72, dtype=float)
        one_hot[target] = 1.0
        confidences.append(float(posterior.max()))
        correct.append(float(int(posterior.argmax()) == target))
        briers.append(float(np.sum((posterior - one_hot) ** 2)))
        nlls.append(float(-np.log(np.clip(posterior[target], 1e-12, 1.0))))
    if not confidences:
        return {"n_sequences": 0, "nll": None, "top_label_ece": None, "multiclass_brier": None}
    conf = np.asarray(confidences)
    acc = np.asarray(correct)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for bin_idx in range(n_bins):
        selected = (conf >= edges[bin_idx]) & (
            conf <= edges[bin_idx + 1] if bin_idx == n_bins - 1 else conf < edges[bin_idx + 1]
        )
        if selected.any():
            ece += float(selected.mean() * abs(acc[selected].mean() - conf[selected].mean()))
    return {"n_sequences": len(conf), "nll": float(np.mean(nlls)),
            "top_label_ece": float(ece), "multiclass_brier": float(np.mean(briers))}


def _fit_temperature(logits, meta, rows):
    if not np.isfinite(logits[rows]).all():
        raise ValueError("Inner calibration scores are incomplete")
    result = minimize_scalar(
        lambda value: _sequence_nll(value, logits, meta, rows),
        bounds=(-2.5, 2.5), method="bounded", options={"xatol": 1e-4},
    )
    if not result.success:
        raise RuntimeError("Q sequence temperature optimization did not converge")
    return float(np.exp(result.x)), float(result.fun)


def _eval_policy(logits, meta, rows, lm_priors=None, *, tau=0.8,
                 min_sequences=2, max_sequences=10):
    true_cells = _true_cells(meta, rows)
    by_char: dict[tuple[str, int], list[int]] = {}
    for index in rows:
        key = (str(meta["run_id"][index]), int(meta["char_idx"][index]))
        if key in true_cells:
            by_char.setdefault(key, []).append(int(index))
    output = {"eeg_only": {"correct": 0, "n": 0, "depth": []},
              "lm_fusion": {"correct": 0, "n": 0, "depth": []}}
    subject_output = {}
    for key, indices in by_char.items():
        ix = np.asarray(indices, dtype=int)
        target = true_cells[key]
        sequences = np.sort(np.unique(meta["seq_idx"][ix]))
        sequences = sequences[sequences >= 0]
        if not len(sequences):
            continue
        cumulative = np.zeros(72, dtype=float)
        eeg_stop = fusion_stop = None
        eeg_guess = fusion_guess = None
        lm_prior = (lm_priors or {}).get(key, np.full(72, 1.0 / 72))
        log_prior = np.full(72, -1e9, dtype=float)
        mapped = lm_prior > 0
        log_prior[mapped] = 0.1 * np.log(np.clip(lm_prior[mapped], 1e-9, 1.0))
        fusion_cumulative = log_prior.copy()
        for depth, sequence in enumerate(sequences[:max_sequences], start=1):
            local = ix[meta["seq_idx"][ix] == sequence]
            posterior = aggregate_membership_logits(
                logits[local], meta["lit_mask"][local], n_rows=9, n_cols=8,
            )
            cumulative += np.log(np.clip(posterior, 1e-12, 1.0))
            fusion_cumulative += np.log(np.clip(posterior, 1e-12, 1.0))
            eeg_post, fusion_post = softmax(cumulative), softmax(fusion_cumulative)
            if depth >= min_sequences and eeg_stop is None and float(eeg_post.max()) >= tau:
                eeg_stop, eeg_guess = depth, int(eeg_post.argmax())
            if depth >= min_sequences and fusion_stop is None and float(fusion_post.max()) >= tau:
                fusion_stop, fusion_guess = depth, int(fusion_post.argmax())
        if eeg_stop is None:
            eeg_stop, eeg_guess = min(max_sequences, len(sequences)), int(softmax(cumulative).argmax())
        if fusion_stop is None:
            fusion_stop, fusion_guess = min(max_sequences, len(sequences)), int(softmax(fusion_cumulative).argmax())
        subject = key[0].split("_SE", 1)[0]
        for name, depth, guess in (("eeg_only", eeg_stop, eeg_guess),
                                   ("lm_fusion", fusion_stop, fusion_guess)):
            stat = output[name]
            stat["n"] += 1
            stat["correct"] += int(guess == target)
            stat["depth"].append(depth)
            per = subject_output.setdefault(subject, {}).setdefault(name, {"n": 0, "correct": 0})
            per["n"] += 1
            per["correct"] += int(guess == target)
    for stat in output.values():
        stat["accuracy"] = stat["correct"] / stat["n"] if stat["n"] else None
        stat["mean_sequences"] = float(np.mean(stat["depth"])) if stat["depth"] else None
    return output, subject_output


def _llm_priors(meta, rows, true_cells, grid, local_files_only=True):
    from src.models.llm_predictor import LLMPredictor

    model = LLMPredictor(grid, local_files_only=local_files_only)
    targets_by_run: dict[str, list[tuple[int, str]]] = {}
    selected_runs = set(meta["run_id"][rows])
    for (run_id, char_idx), target in true_cells.items():
        if run_id in selected_runs:
            targets_by_run.setdefault(run_id, []).append((char_idx, model.char_list[target]))
    priors = {}
    context_cache = {}
    for run_id, sequence in targets_by_run.items():
        sequence.sort()
        context = ""
        for char_idx, symbol in sequence:
            key = (run_id, char_idx)
            if context not in context_cache:
                context_cache[context] = model.predict_next_char(context)
            priors[key] = context_cache[context]
            context += model.special_key_map.get(symbol, str(symbol).lower())
    return priors


def run(cache_path: Path, output_path: Path, score_path: Path | None = None,
        *, outer_folds: int = 3, inner_folds: int = 2, with_lm: bool = True) -> dict:
    with h5py.File(cache_path, "r") as h5:
        meta = {
            "y": h5["y"][:].astype(np.uint8),
            "run_id": _decode(h5["run_id"][:]),
            "subject": _decode(h5["subject"][:]),
            "session": _decode(h5["session"][:]),
            "char_idx": h5["char_idx"][:].astype(np.int32),
            "seq_idx": h5["seq_idx"][:].astype(np.int32),
            "lit_mask": h5["lit_mask"][:].astype(np.uint8),
        }
        all_rows = np.arange(len(meta["y"]))
        heldout = {r for values in json.loads(MANIFEST.read_text()).values() for r in values}
        excluded = {r for values in json.loads(EXCLUSIONS.read_text()).values() for r in values}
        unique_runs = sorted(set(meta["run_id"]))
        assert_training_pool_disjoint([f"{r}-epo.fif" for r in unique_runs], heldout)
        if set(unique_runs) & excluded or any("_SE003_" in r or "_Test" in r for r in unique_runs):
            raise ValueError("Q calibration/fusion cache is not disjoint from held-out data")
        groups = _groups(meta)
        outer = GroupKFold(n_splits=min(outer_folds, len(set(groups))))
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        calibrated = np.full(len(all_rows), np.nan, dtype=float)
        raw = np.full(len(all_rows), np.nan, dtype=float)
        temperatures, folds = [], []
        for outer_id, (train_idx, test_idx) in enumerate(
            outer.split(all_rows, meta["y"], groups=groups), start=1
        ):
            calibration = np.full(len(all_rows), np.nan, dtype=float)
            inner = GroupKFold(n_splits=min(inner_folds, len(set(groups[train_idx]))))
            for inner_id, (fit_pos, cal_pos) in enumerate(
                inner.split(train_idx, meta["y"][train_idx], groups=groups[train_idx]), start=1
            ):
                fit_idx, cal_idx = train_idx[fit_pos], train_idx[cal_pos]
                _, _, scores, pred_idx, _ = _fit_m0_streaming(
                    h5["X"], fit_idx, cal_idx, meta["y"], M0_PREPROCESSING,
                    channels, sfreq, tmin, seed=2000 + 10 * outer_id + inner_id,
                )
                calibration[pred_idx] = scores
                print(f"Q nested calibration outer {outer_id} inner {inner_id} complete", flush=True)
            calibration_idx = train_idx[np.isfinite(calibration[train_idx])]
            temperature, calibration_nll = _fit_temperature(calibration, meta, calibration_idx)
            _, _, test_scores, pred_idx, _ = _fit_m0_streaming(
                h5["X"], train_idx, test_idx, meta["y"], M0_PREPROCESSING,
                channels, sfreq, tmin, seed=3000 + outer_id,
            )
            raw[pred_idx] = test_scores
            calibrated[pred_idx] = test_scores / temperature
            outer_raw_metrics = _sequence_metrics(raw, meta, pred_idx)
            outer_calibrated_metrics = _sequence_metrics(calibrated, meta, pred_idx)
            temperatures.append({"outer_fold": outer_id, "temperature": temperature,
                                 "inner_sequence_nll": calibration_nll,
                                 "outer_raw_sequence_metrics": outer_raw_metrics,
                                 "outer_calibrated_sequence_metrics": outer_calibrated_metrics})
            folds.append({"outer_fold": outer_id,
                          "validation_sessions": sorted(set(groups[test_idx])),
                          "calibration_sessions": sorted(set(groups[calibration_idx])),
                          "overlap": bool(set(groups[test_idx]) & set(groups[calibration_idx]))})
            print(f"Q nested calibrated outer fold {outer_id} complete (T={temperature:.4f})", flush=True)

        valid = np.isfinite(calibrated)
        true_cells = _true_cells(meta, np.flatnonzero(valid))
        layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())["StudyQ"]
        grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
        for symbol, (row, col) in layout["grid_map"].items():
            grid[row - 1, col - 1] = symbol
        priors = _llm_priors(meta, np.flatnonzero(valid), true_cells, grid) if with_lm else {}
        metrics, per_subject = _eval_policy(calibrated, meta, np.flatnonzero(valid), priors)
        result = {
            "study": "StudyQ", "model": "M0_clean_SGD",
            "outer_grouping": "subject-session", "outer_folds": len(folds),
            "inner_calibration_grouping": "subject-session", "inner_folds": inner_folds,
            "calibration": "nested inner OOF scalar temperature; sequence-level membership-grid NLL",
            "temperatures": temperatures, "folds": folds,
            "fusion": {"kind": "local DistilGPT2 prior" if with_lm else "disabled", "alpha": 0.1 if with_lm else 0.0,
                       "applied_once_before_sequence_accumulation": True},
            "stopping": {"tau": 0.8, "min_sequences": 2, "max_sequences": 10},
            "heldout_session": "SE003", "heldout_test_data_read": False,
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "metrics": metrics, "per_subject": per_subject,
        }
        if score_path:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(score_path, logits=calibrated, raw_logits=raw,
                                run_id=meta["run_id"], subject=meta["subject"],
                                session=meta["session"], char_idx=meta["char_idx"],
                                seq_idx=meta["seq_idx"], y=meta["y"], lit_mask=meta["lit_mask"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/study_q_classifier_epochs.h5")
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--no-lm", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/study_q_nested_fusion.json")
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/study_q_nested_fusion_scores.npz")
    args = parser.parse_args()
    result = run(args.cache, args.output, args.scores, outer_folds=args.outer_folds,
                 inner_folds=args.inner_folds, with_lm=not args.no_lm)
    print(json.dumps({"metrics": result["metrics"], "temperatures": result["temperatures"]}, indent=2))
