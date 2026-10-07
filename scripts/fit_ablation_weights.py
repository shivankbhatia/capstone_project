#!/usr/bin/env python3
"""Fit rung fusion weights on a deterministic, run-grouped OOF split only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.special import log_softmax, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import load_spelling_matrix  # noqa: E402
from scripts.build_phrase_bank_from_registry import is_test  # noqa: E402
from scripts.evaluate_inner_fusion import _decode_strings  # noqa: E402
from scripts.screen_shrinkage_lda import _load_metadata  # noqa: E402
from src.models.llm_predictor import LLMPredictor  # noqa: E402
from src.models.rag_predictor import RAGPredictor  # noqa: E402


class _FixedPrior:
    def __init__(self, char_list, probabilities):
        self.char_list = char_list
        self.probabilities = probabilities

    def predict_next_char(self, _context):
        return self.probabilities


def _targets_and_sequences(meta, logits):
    run_ids = meta["run_id"]
    chars, seqs = meta["char_idx"], meta["seq_idx"]
    y, codes = meta["y"], meta["stim_code"]
    targets = defaultdict(lambda: defaultdict(int))
    sequences = defaultdict(list)
    for idx, (run_id, char_idx, seq_idx) in enumerate(zip(run_ids, chars, seqs)):
        if char_idx < 0:
            continue
        if y[idx] == 1:
            targets[(str(run_id), int(char_idx))][int(codes[idx])] += 1
    by_sequence = defaultdict(list)
    for idx, (run_id, char_idx, seq_idx) in enumerate(zip(run_ids, chars, seqs)):
        if char_idx >= 0 and seq_idx > 0 and np.isfinite(logits[idx]):
            by_sequence[(str(run_id), int(char_idx), int(seq_idx))].append(idx)
    for (run_id, char_idx, seq_idx), indices in by_sequence.items():
        idx = np.asarray(indices, dtype=int)
        row = np.zeros(9)
        col = np.zeros(8)
        for score, code in zip(logits[idx], codes[idx]):
            if 1 <= code <= 9:
                row[int(code) - 1] += score
            elif 10 <= code <= 17:
                col[int(code) - 10] += score
        target_codes = codes[idx][y[idx] == 1]
        tr = target_codes[(target_codes >= 1) & (target_codes <= 9)]
        tc = target_codes[(target_codes >= 10) & (target_codes <= 17)]
        if not len(tr) or not len(tc):
            continue
        sequences[(run_id, char_idx)].append((
            seq_idx, log_softmax(row), log_softmax(col),
            int(np.bincount(tr - 1).argmax()) * 8 + int(np.bincount(tc - 10).argmax()),
        ))
    targets_idx = {}
    for key, counts in targets.items():
        rows = {code: count for code, count in counts.items() if 1 <= code <= 9}
        cols = {code: count for code, count in counts.items() if 10 <= code <= 17}
        if rows and cols:
            targets_idx[key] = (max(rows, key=rows.get) - 1) * 8 + max(cols, key=cols.get) - 10
    for key, values in sequences.items():
        values.sort(key=lambda item: item[0])
    return targets_idx, sequences


def _nll_for_alpha(char_keys, sequences, priors, alpha, adaptive, max_sequences=10):
    losses = []
    for key in char_keys:
        if key not in sequences or key not in priors:
            continue
        seqs = sequences[key][:max_sequences]
        if not seqs:
            continue
        prior = np.clip(np.asarray(priors[key], dtype=float), 1e-12, 1.0)
        if adaptive:
            entropy = -np.sum(prior * np.log(prior))
            alpha_eff = alpha * (1.0 - entropy / np.log(len(prior)))
        else:
            alpha_eff = alpha
        accumulated = alpha_eff * np.log(prior)
        target = seqs[0][3]
        for _, row_logp, col_logp, sequence_target in seqs:
            if sequence_target != target:
                continue
            accumulated += (row_logp[:, None] + col_logp[None, :]).reshape(-1)
        losses.append(-float(log_softmax(accumulated)[target]))
    return float(np.mean(losses)) if losses else float("inf")


def _fit_alpha(char_keys, sequences, priors, adaptive):
    grid = np.linspace(0.0, 5.0, 101)
    scored = [(_nll_for_alpha(char_keys, sequences, priors, alpha, adaptive), float(alpha))
              for alpha in grid]
    nll, alpha = min(scored)
    return {"alpha": alpha, "oof_sequence_nll": nll, "grid": grid.tolist(),
            "adaptive": adaptive}


def run(cache_path: Path, scores_path: Path, output_path: Path):
    spelling, _, _ = load_spelling_matrix("data/processed/grid_layout.json", study_name="StudyD")
    char_list = list(np.asarray(spelling).ravel())
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
    meta["run_id"] = _decode_strings(meta["run_id"])
    meta["subject"] = _decode_strings(meta["subject"])
    with np.load(scores_path, allow_pickle=False) as stored:
        logits = stored["M0_calibrated"].astype(float)
    if not np.isfinite(logits).any():
        raise RuntimeError("M0 OOF sidecar contains no finite calibrated scores")

    target_cells, sequences = _targets_and_sequences(meta, logits)
    target_chars = {key: char_list[cell] for key, cell in target_cells.items()}
    validation_runs = {run_id for run_id in set(meta["run_id"]) if is_test(str(run_id))}
    if not validation_runs:
        raise RuntimeError("The deterministic 20% OOF fusion-fit split is empty")
    fit_keys = sorted(key for key in target_chars if key[0] in validation_runs)
    if not fit_keys:
        raise RuntimeError("No OOF character targets matched the fusion-fit run split")

    llm = LLMPredictor(spelling, local_files_only=True)
    contexts = {}
    for run_id in sorted(set(meta["run_id"])):
        chars = {char: target_chars[(run_id, char)] for rid, char in target_chars if rid == run_id}
        for position, char_idx in enumerate(sorted(chars)):
            context = "".join(chars[idx] for idx in sorted(chars) if idx < char_idx)
            contexts[(run_id, char_idx)] = context
    fit_contexts = {key: contexts[key] for key in fit_keys}
    base_priors = {key: llm.predict_next_char(context) for key, context in fit_contexts.items()}
    subjects = {run_id: str(run_id).split("_SE", 1)[0] for run_id, _ in fit_contexts}

    base_fit = {key: base_priors[key] for key in fit_keys}
    weights = {"fit_scope": "20% deterministic SHA-256 run split from training-pool OOF only",
               "validation_runs": sorted(validation_runs), "validation_characters": len(fit_keys),
               "rung_1": _fit_alpha(fit_keys, sequences, base_fit, adaptive=False),
               "rag": {}}

    rag_weight_grid = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))
    retrieval_threshold_grid = (0.0, 0.2, 0.4, 0.6, 0.8)
    rag_rungs = ("rung_2_global", "rung_3_subject_only", "rung_4_personalized")
    for rung in rag_rungs:
        best = None
        thresholds = retrieval_threshold_grid if rung != "rung_3_subject_only" else (0.6,)
        for retrieval_threshold in thresholds:
            for rag_weight in rag_weight_grid:
                wrappers = {}
                for subject in sorted(set(subjects.values())):
                    wrapper = RAGPredictor(
                        _FixedPrior(char_list, np.full(len(char_list), 1.0 / len(char_list))),
                        phrase_bank_path="data/rag/phrase_bank_global.csv",
                        rag_weight=rag_weight,
                        retrieval_confidence_threshold=retrieval_threshold,
                        subject_id=subject if rung != "rung_2_global" else None,
                        min_subject_phrases=10,
                        personalization_bonus=1.5,
                        subject_only=(rung == "rung_3_subject_only"),
                        subject_confidence_threshold=0.20,
                        sufficiency_midpoint_tokens=15.0,
                        sufficiency_sharpness=0.20,
                        subject_gate_sharpness=10.0,
                        bigram_min_count=2,
                        bigram_max_normalized_entropy=0.65,
                    )
                    wrappers[subject] = wrapper
                priors = {}
                for key, context in fit_contexts.items():
                    run_id, _ = key
                    wrapper = wrappers[subjects[run_id]]
                    wrapper.base_predictor.probabilities = base_priors[key]
                    priors[key] = wrapper.predict_next_char(context)
                result = _fit_alpha(fit_keys, sequences, priors, adaptive=True)
                candidate = {"rag_weight": rag_weight,
                             "retrieval_confidence_threshold": retrieval_threshold,
                             **result}
                if best is None or candidate["oof_sequence_nll"] < best["oof_sequence_nll"]:
                    best = candidate
        weights["rag"][rung] = best
        print(f"{rung}: alpha={best['alpha']:.3f} rag_weight={best['rag_weight']:.2f} "
              f"OOF NLL={best['oof_sequence_nll']:.4f}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline.h5")
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/classifier_nested_calibration_scores.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/ablation_oof_weights.json")
    args = parser.parse_args()
    result = run(args.cache, args.scores, args.output)
    print(json.dumps({key: val if key != "grid" else "<41-point grid>"
                      for key, val in result.items() if key != "rag"}, indent=2))


if __name__ == "__main__":
    main()
