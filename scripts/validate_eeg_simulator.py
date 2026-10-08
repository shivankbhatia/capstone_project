#!/usr/bin/env python3
"""LOSO validation of an i.i.d. empirical flash-logit simulator on clean D/Q OOF.

All inputs are training-pool OOF artifacts. No held-out registry or manifest
labels are imported. Outputs are simulations, never measured EEG results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nested_study_q_calibration_fusion import _llm_priors as q_llm_priors
from scripts.nested_study_q_calibration_fusion import _true_cells as q_true_cells
from src.evaluation.sequence_scoring import aggregate_flash_logits, aggregate_membership_logits
from src.models.llm_predictor import LLMPredictor


def _strs(array):
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in array])


def _grid(study):
    layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())[study]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for symbol, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = symbol
    return grid


def _d_true_cells(meta, rows):
    grouped = {}
    for i in rows:
        key = (str(meta["run_id"][i]), int(meta["char_idx"][i]))
        if key[1] >= 0:
            grouped.setdefault(key, []).append(int(i))
    truth = {}
    for key, indices in grouped.items():
        ix = np.asarray(indices)
        codes = meta["stim_code"][ix][meta["y"][ix] == 1]
        row_codes = codes[(codes >= 1) & (codes <= 9)]
        col_codes = codes[(codes >= 10) & (codes <= 17)]
        if not len(row_codes) or not len(col_codes):
            continue
        row = int(np.bincount(row_codes - 1, minlength=9).argmax())
        col = int(np.bincount(col_codes - 10, minlength=8).argmax())
        truth[key] = row * 8 + col
    return truth


def _priors(meta, truth, grid, rows):
    model = LLMPredictor(grid, local_files_only=True)
    selected = set(meta["run_id"][rows])
    targets = {}
    for (run_id, char_idx), cell in truth.items():
        if run_id in selected:
            targets.setdefault(run_id, []).append((char_idx, str(grid.reshape(-1)[cell])))
    priors = {}
    for run_id, chars in targets.items():
        context = ""
        for char_idx, symbol in sorted(chars):
            priors[(run_id, char_idx)] = model.predict_next_char(context)
            context += model.special_key_map.get(symbol, str(symbol).lower())
    return priors


def _prepare(study):
    if study == "Q":
        cache = ROOT / "data/cache/study_q_classifier_epochs.h5"
        with h5py.File(cache, "r") as h5:
            meta = {name: h5[name][:] for name in
                    ("run_id", "subject", "char_idx", "seq_idx", "y", "lit_mask")}
        meta["run_id"], meta["subject"] = _strs(meta["run_id"]), _strs(meta["subject"])
        with np.load(ROOT / "results/tables/study_q_nested_fusion_scores.npz", allow_pickle=False) as scores:
            logits = scores["logits"]
        rows = np.flatnonzero(np.isfinite(logits))
        truth = q_true_cells(meta, rows)
        grid = _grid("StudyQ")
        priors = q_llm_priors(meta, rows, truth, grid)
        meta["lit_mask"] = meta["lit_mask"].astype(np.uint8)
        return {"study": study, "meta": meta, "logits": logits, "rows": rows,
                "truth": truth, "grid": grid, "priors": priors, "kind": "group"}
    cache = ROOT / "data/cache/classifier_1_12.h5"
    with h5py.File(cache, "r") as h5:
        meta = {name: h5[name][:] for name in
                ("run_id", "subject", "char_idx", "seq_idx", "stim_code", "y")}
    meta["run_id"], meta["subject"] = _strs(meta["run_id"]), _strs(meta["subject"])
    with np.load(ROOT / "results/tables/classifier_nested_calibration_scores.npz", allow_pickle=False) as scores:
        logits = scores["M0_calibrated"]
    rows = np.flatnonzero(np.isfinite(logits))
    truth = _d_true_cells(meta, rows)
    grid = _grid("StudyD")
    priors = _priors(meta, truth, grid, rows)
    return {"study": study, "meta": meta, "logits": logits, "rows": rows,
            "truth": truth, "grid": grid, "priors": priors, "kind": "rowcol"}


def _char_sequences(data):
    meta = data["meta"]
    by_char, by_seq = {}, {}
    for index in data["rows"]:
        run_id, char_idx = str(meta["run_id"][index]), int(meta["char_idx"][index])
        if char_idx < 0 or (run_id, char_idx) not in data["truth"]:
            continue
        key = (run_id, char_idx)
        by_char.setdefault(key, []).append(int(index))
        sequence = int(meta["seq_idx"][index])
        if sequence >= 0:
            by_seq.setdefault(key + (sequence,), []).append(int(index))
    output = {}
    for key, indices in by_char.items():
        sequences = sorted(seq for run_id, char_idx, seq in by_seq if (run_id, char_idx) == key)
        output[key] = [(seq, np.asarray(by_seq[key + (seq,)], dtype=int)) for seq in sequences[:10]]
    return output


def _sequence_posterior(data, logits, indices):
    meta = data["meta"]
    if data["kind"] == "group":
        return aggregate_membership_logits(logits[indices], meta["lit_mask"][indices], 9, 8)
    return aggregate_flash_logits(logits[indices], meta["stim_code"][indices], 9, 8).grid


def _curve(data, chars, subjects, rng=None, pools=None, replicates=1):
    """Return accuracy-vs-depth and fixed-policy stopping accuracy."""
    meta, logits = data["meta"], data["logits"]
    subject_curve_correct = {s: np.zeros(10) for s in subjects}
    subject_curve_n = {s: np.zeros(10) for s in subjects}
    subject_stop_correct = {s: 0 for s in subjects}
    subject_stop_n = {s: 0 for s in subjects}
    for (run_id, char_idx), sequences in chars.items():
        subject = str(meta["subject"][sequences[0][1][0]])
        if subject not in subjects:
            continue
        target = int(data["truth"][(run_id, char_idx)])
        prior = np.asarray(data["priors"].get((run_id, char_idx), np.ones(72) / 72), dtype=float)
        log_prior = np.full(72, -1e9)
        log_prior[prior > 0] = 0.1 * np.log(np.clip(prior[prior > 0], 1e-12, 1.0))
        max_depth = min(10, len(sequences))
        for _ in range(replicates):
            cumulative = np.zeros(72)
            fused = log_prior.copy()
            stopped = False
            stop_guess = None
            for depth, (_, ix) in enumerate(sequences[:10], start=1):
                scores = logits
                if rng is not None:
                    simulated = np.empty(len(ix), dtype=float)
                    labels = meta["y"][ix]
                    for cls in (0, 1):
                        positions = np.flatnonzero(labels == cls)
                        pool = pools[cls]
                        simulated[positions] = rng.choice(pool, size=len(positions), replace=True)
                    if data["kind"] == "group":
                        post = aggregate_membership_logits(simulated, meta["lit_mask"][ix], 9, 8)
                    else:
                        post = aggregate_flash_logits(simulated, meta["stim_code"][ix], 9, 8).grid
                else:
                    post = _sequence_posterior(data, logits, ix)
                evidence = np.log(np.clip(post, 1e-12, 1.0))
                cumulative += evidence
                fused += evidence
                guess = int(softmax(fused).argmax())
                if not stopped and depth >= 2 and float(softmax(fused).max()) >= 0.8:
                    stopped, stop_guess = True, guess
                if depth == max_depth:
                    if stop_guess is None:
                        stop_guess = int(softmax(fused).argmax())
                    subject_stop_correct[subject] += int(stop_guess == target)
                    subject_stop_n[subject] += 1
                if rng is None:
                    if int(softmax(fused).argmax()) == target:
                        subject_curve_correct[subject][depth - 1] += 1
                    subject_curve_n[subject][depth - 1] += 1
                else:
                    subject_curve_correct[subject][depth - 1] += int(int(softmax(fused).argmax()) == target)
                    subject_curve_n[subject][depth - 1] += 1
    curves = {}
    for subject in subjects:
        curves[subject] = [float(c / n) if n else None for c, n in
                           zip(subject_curve_correct[subject], subject_curve_n[subject])]
    return curves, subject_stop_correct, subject_stop_n


def validate_study(study, *, replicates=20, seed=20261008):
    data = _prepare(study)
    chars = _char_sequences(data)
    subjects = sorted(set(data["meta"]["subject"][data["rows"]]))
    # Observed OOF performance curve and accuracy with the locked stopping rule.
    observed_curves, observed_stop, observed_n = _curve(data, chars, subjects)
    observed = [float(np.mean([v[d] for v in observed_curves.values() if v[d] is not None]))
                for d in range(10)]
    observed_stop_accuracy = sum(observed_stop.values()) / max(1, sum(observed_n.values()))

    # Rotating LOSO: score distributions exclude the validation subject.
    simulated_subject_curves = {s: np.zeros(10) for s in subjects}
    simulated_subject_counts = {s: np.zeros(10) for s in subjects}
    simulated_stop_correct = simulated_stop_n = 0
    rng = np.random.default_rng(seed)
    y = data["meta"]["y"]
    for held_subject in subjects:
        train_mask = np.isin(data["meta"]["subject"], [s for s in subjects if s != held_subject])
        train_rows = data["rows"][train_mask[data["rows"]]]
        pools = {cls: data["logits"][train_rows[(y[train_rows] == cls)]] for cls in (0, 1)}
        if any(len(pool) < 10 for pool in pools.values()):
            raise ValueError(f"Insufficient LOSO training scores for {study} {held_subject}")
        target_chars = {k: v for k, v in chars.items()
                        if str(data["meta"]["subject"][v[0][1][0]]) == held_subject}
        for _ in range(replicates):
            curve, stop_correct, stop_n = _curve(
                data, target_chars, [held_subject], rng=rng, pools=pools, replicates=1,
            )
            for depth, value in enumerate(curve[held_subject]):
                if value is not None:
                    # For one replication the curve entries are accuracies; average replicates below.
                    simulated_subject_curves[held_subject][depth] += value
                    simulated_subject_counts[held_subject][depth] += 1
            simulated_stop_correct += sum(stop_correct.values())
            simulated_stop_n += sum(stop_n.values())
    simulated_curves = {}
    for subject in subjects:
        simulated_curves[subject] = [
            float(v / n) if n else None
            for v, n in zip(simulated_subject_curves[subject], simulated_subject_counts[subject])
        ]
    actual_curve = [float(np.mean([v[d] for v in observed_curves.values() if v[d] is not None]))
                    for d in range(10)]
    simulated_curve = [float(np.mean([v[d] for v in simulated_curves.values() if v[d] is not None]))
                       for d in range(10)]
    errors = [abs(a - b) for a, b in zip(actual_curve, simulated_curve)]
    mae, max_error = float(np.mean(errors)), float(np.max(errors))
    tolerance = json.loads((ROOT / "splits/scheduler_lock.json").read_text())["simulator_validation"]["leave_subject_out_curve"]
    known = {"D": 0.7581, "Q": 0.6875}[study]
    reference_error = abs(simulated_stop_correct / max(1, simulated_stop_n) - known)
    return {
        "study": study, "simulated": True,
        "input_scope": "clean training-pool nested OOF flash logits only",
        "subjects": len(subjects), "characters": len(chars), "replicates": replicates,
        "accuracy_vs_depth": {str(d + 1): {"observed_oof": actual_curve[d], "simulated_loso": simulated_curve[d], "absolute_error": errors[d]}
                               for d in range(10)},
        "observed_oof_fixed_tau_0p8_accuracy": observed_stop_accuracy,
        "simulated_loso_fixed_tau_0p8_accuracy": simulated_stop_correct / max(1, simulated_stop_n),
        "known_result_reference_accuracy": known,
        "known_result_absolute_error": reference_error,
        "known_result_validation_pass": reference_error <= 0.05,
        "curve_mae": mae, "curve_max_pointwise_error": max_error,
        "curve_validation_pass": mae <= 0.10 and max_error <= 0.15,
        "limitations": ["i.i.d. score sampling ignores within-character temporal dependence, adaptation, fatigue, and nonstationary artifacts."],
    }


if __name__ == "__main__":
    outputs = {study: validate_study(study) for study in ("D", "Q")}
    result = {"simulated": True, "label": "simulated; not measured EEG",
              "seed": 20261008, "outputs": outputs}
    path = ROOT / "results/tables/eeg_simulator_loso_validation.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in x.items() if k not in {"accuracy_vs_depth", "limitations"}}
                      for s, x in outputs.items()}, indent=2))
