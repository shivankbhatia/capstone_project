#!/usr/bin/env python3
"""Evaluate nested-calibrated OOF sequences with the existing LM prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import load_spelling_matrix  # noqa: E402
from src.models.decoder import calculate_itr  # noqa: E402
from src.models.fusion import BayesianFusionEngine  # noqa: E402
from src.models.llm_predictor import LLMPredictor  # noqa: E402
from scripts.screen_shrinkage_lda import _load_metadata, _decode_strings  # noqa: E402


def _grid_sequence(logits, labels, codes):
    rows, cols = np.zeros(9), np.zeros(8)
    target_codes = codes[labels == 1]
    true_row = target_codes[(target_codes >= 1) & (target_codes <= 9)]
    true_col = target_codes[(target_codes >= 10) & (target_codes <= 17)]
    if not len(true_row) or not len(true_col):
        return None
    for logit, code in zip(logits, codes):
        if 1 <= code <= 9:
            rows[code - 1] += logit
        elif 10 <= code <= 17:
            cols[code - 10] += logit
    cell = np.zeros((9, 8))
    cell[:] = rows[:, None] + cols[None, :]
    true_r = int(np.bincount(true_row - 1).argmax())
    true_c = int(np.bincount(true_col - 10).argmax())
    return softmax(cell.ravel()), softmax(rows), softmax(cols), true_r, true_c


def _ece(confidences, accuracies, bins=15):
    confidences = np.asarray(confidences)
    accuracies = np.asarray(accuracies)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences >= lo) & ((confidences < hi) if hi < 1 else (confidences <= hi))
        if mask.any():
            value += float(mask.mean()) * abs(float(confidences[mask].mean() - accuracies[mask].mean()))
    return value


def _decode_model(logits, meta, char_list, llm_priors, tau=0.80, min_seq=2, max_seq=10,
                  alpha=0.0):
    groups = {}
    for i, key in enumerate(zip(meta["run_id"], meta["char_idx"], meta["seq_idx"])):
        run, char, seq = str(key[0]), int(key[1]), int(key[2])
        if char >= 0 and seq >= 0:
            groups.setdefault((run, char, seq), []).append(i)

    by_char = {}
    sequence_nll = {}
    sequence_calibration = {}
    for (run, char, seq), rows in groups.items():
        ix = np.asarray(rows, dtype=int)
        if len(ix) != 17 or not np.isfinite(logits[ix]).all():
            continue
        result = _grid_sequence(logits[ix], meta["y"][ix], meta["stim_code"][ix])
        if result is None:
            continue
        grid_p, row_p, col_p, true_r, true_c = result
        cell_idx = true_r * 8 + true_c
        sequence_nll[(run, char, seq)] = (
            -np.log(max(row_p[true_r], 1e-15)), -np.log(max(col_p[true_c], 1e-15))
        )
        sequence_calibration[(run, char, seq)] = (
            (float(row_p.max()), int(row_p.argmax() == true_r)),
            (float(col_p.max()), int(col_p.argmax() == true_c)),
        )
        by_char.setdefault((run, char), []).append((seq, grid_p, cell_idx, true_r, true_c))

    char_results = []
    for (run, char), seqs in by_char.items():
        seqs.sort(key=lambda item: item[0])
        target_row = int(np.bincount([item[3] for item in seqs]).argmax())
        target_col = int(np.bincount([item[4] for item in seqs]).argmax())
        target = target_row * 8 + target_col
        prior = np.asarray(llm_priors[(run, char)], dtype=float)
        if alpha:
            engine = BayesianFusionEngine(len(char_list), mode="fixed", base_alpha=alpha)
            initial = engine.get_initial_log_bias(prior)
        else:
            initial = np.zeros(len(char_list), dtype=float)
        accumulated = initial.copy()
        stopped_prediction, stopped_at = None, None
        for step, (_, p, _, _, _) in enumerate(seqs[:max_seq], start=1):
            accumulated += np.log(np.clip(p, 1e-9, 1.0))
            post = softmax(accumulated)
            prediction = int(post.argmax())
            if step >= min_seq and post.max() >= tau:
                stopped_prediction, stopped_at = prediction, step
                break
        if stopped_prediction is None:
            # Match the locked policy's max-sequence fallback.
            stopped_at = min(len(seqs), max_seq)
            used = seqs[:stopped_at]
            accumulated = initial.copy()
            for _, p, _, _, _ in used:
                accumulated += np.log(np.clip(p, 1e-9, 1.0))
            stopped_prediction = int(softmax(accumulated).argmax())
        full_acc = initial.copy()
        for _, p, _, _, _ in seqs[:max_seq]:
            full_acc += np.log(np.clip(p, 1e-9, 1.0))
        char_results.append({
            "subject": run.split("_SE", 1)[0], "run_id": run,
            "char_idx": char, "target_idx": target,
            "predicted_idx": int(stopped_prediction),
            "stopped_correct": int(stopped_prediction == target),
            "full_correct": int(int(full_acc.argmax()) == target),
            "sequences_used": int(stopped_at),
            "char_label": str(char_list[target]),
        })
    return char_results, sequence_nll, sequence_calibration


def _per_subject(char_results):
    output = {}
    for row in char_results:
        d = output.setdefault(row["subject"], {"n": 0, "stopped_correct": 0,
                                                  "full_correct": 0, "sequences": []})
        d["n"] += 1
        d["stopped_correct"] += row["stopped_correct"]
        d["full_correct"] += row["full_correct"]
        d["sequences"].append(row["sequences_used"])
    for d in output.values():
        d["stopped_accuracy"] = d["stopped_correct"] / d["n"]
        d["max_sequence_accuracy"] = d["full_correct"] / d["n"]
        d["mean_sequences"] = float(np.mean(d["sequences"]))
        del d["sequences"]
    return output


def _summarize(char_results, sequence_nll, sequence_calibration):
    subjects = _per_subject(char_results)
    n = len(char_results)
    confidence_pairs = [x for pair in sequence_calibration.values() for x in pair]
    smooth_by_subject = {}
    for (run_id, _, _), pair in sequence_nll.items():
        smooth_by_subject.setdefault(run_id.split("_SE", 1)[0], []).extend(pair)
    for subject, values in smooth_by_subject.items():
        subjects.setdefault(subject, {})["sequence_row_column_nll"] = float(np.mean(values))
    stopped_accuracy = (
        sum(x["stopped_correct"] for x in char_results) / n if n else None
    )
    mean_sequences = (
        float(np.mean([x["sequences_used"] for x in char_results])) if n else None
    )
    itr = calculate_itr(72, stopped_accuracy * 100, mean_sequences * 2.0) if n else None
    return {
        "characters": n,
        "stopped_accuracy": stopped_accuracy,
        "max_sequence_accuracy": sum(x["full_correct"] for x in char_results) / n if n else None,
        "mean_sequences": mean_sequences,
        "itr_bits_per_minute": itr,
        "wpm_approx": itr / 5.0 if itr is not None else None,
        "sequence_row_column_nll": float(np.mean([v for pair in sequence_nll.values() for v in pair])) if sequence_nll else None,
        "sequence_top_label_ece": _ece(
            [x[0] for x in confidence_pairs], [x[1] for x in confidence_pairs]
        ) if confidence_pairs else None,
        "per_subject": subjects,
    }


def run(scores_path, cache_path, study="StudyD", alpha=0.1, tau=0.8,
        min_seq=2, max_seq=10, output_path=None, policy_sweep=False,
        policy_models=None):
    matrix, _, _ = load_spelling_matrix("data/processed/grid_layout.json", study_name=study)
    llm = LLMPredictor(matrix, local_files_only=True)
    char_list = llm.char_list
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
    meta["run_id"] = _decode_strings(meta["run_id"])
    meta["subject"] = _decode_strings(meta["subject"])
    with np.load(scores_path, allow_pickle=False) as stored:
        score_names = list(stored.files)
        models = {}
        for name in score_names:
            if name.endswith("_calibrated"):
                models[name.removesuffix("_calibrated")] = stored[name].astype(float)
    # Reconstruct character targets from clean-pool epoch labels. No registry
    # or evaluation labels are loaded by this script.
    target_by_char = {}
    for i, key in enumerate(zip(meta["run_id"], meta["char_idx"])):
        run, char = str(key[0]), int(key[1])
        if char < 0 or not meta["y"][i]:
            continue
        codes = target_by_char.setdefault((run, char), {})
        code = int(meta["stim_code"][i])
        codes[code] = codes.get(code, 0) + 1
    target_chars = {}
    for key, counts in target_by_char.items():
        rows = [c for c in counts if 1 <= c <= 9]
        cols = [c for c in counts if 10 <= c <= 17]
        if rows and cols:
            row = max(rows, key=lambda code: counts[code])
            col = max(cols, key=lambda code: counts[code])
            r, c = row - 1, col - 10
            target_chars[key] = int(r * 8 + c)

    llm_priors = {}
    per_run = {}
    for run, char in target_chars:
        per_run.setdefault(run, {})[char] = char_list[target_chars[(run, char)]]
    for run, chars in per_run.items():
        ordered = sorted(chars)
        for pos, char_idx in enumerate(ordered):
            context = "".join(chars[c] for c in ordered[:pos])
            llm_priors[(run, char_idx)] = llm.predict_next_char(context)

    output = {
        "cv": "nested run-grouped OOF; calibration fitted on inner OOF of each outer training pool",
        "fusion": {"implementation": "BayesianFusionEngine initial prior", "mode": "fixed", "base_alpha": alpha},
        "stopping": {"tau": tau, "min_sequences": min_seq, "max_sequences": max_seq},
        "models": {},
    }
    for model_name, logits in models.items():
        eeg, eeg_nll, eeg_cal = _decode_model(logits, meta, char_list, llm_priors,
                                              tau=tau, min_seq=min_seq, max_seq=max_seq, alpha=0.0)
        fused, fused_nll, fused_cal = _decode_model(logits, meta, char_list, llm_priors,
                                                    tau=tau, min_seq=min_seq, max_seq=max_seq, alpha=alpha)
        output["models"][model_name] = {
            "eeg_only": _summarize(eeg, eeg_nll, eeg_cal),
            "fusion": _summarize(fused, fused_nll, fused_cal),
        }
    if policy_sweep:
        policy_models = list(policy_models or models)
        unknown_policy_models = set(policy_models) - set(models)
        if unknown_policy_models:
            raise ValueError(f"Unknown policy model(s): {sorted(unknown_policy_models)}")
        grid = []
        default_predictions = {}
        for model_name in policy_models:
            logits = models[model_name]
            default_rows, _, _ = _decode_model(
                logits, meta, char_list, llm_priors,
                tau=0.80, min_seq=2, max_seq=10, alpha=alpha,
            )
            default_predictions[model_name] = {
                (row["run_id"], row["char_idx"]): row
                for row in default_rows
            }
        # Tune from inner OOF sequences only. Selection is by fused stopped
        # character accuracy; ties prefer fewer flashes, then higher ITR.
        for min_count in (2, 3):
            for max_count in (5, 8, 10):
                if min_count > max_count:
                    continue
                for threshold in (0.60, 0.70, 0.80, 0.90):
                    for model_name in policy_models:
                        logits = models[model_name]
                        fused, nll, cal = _decode_model(
                            logits, meta, char_list, llm_priors,
                            tau=threshold, min_seq=min_count,
                            max_seq=max_count, alpha=alpha,
                        )
                        summary = _summarize(fused, nll, cal)
                        candidate_predictions = {
                            (row["run_id"], row["char_idx"]): row
                            for row in fused
                        }
                        shared_chars = set(default_predictions[model_name]) & set(candidate_predictions)
                        changed = [key for key in shared_chars
                                   if default_predictions[model_name][key]["predicted_idx"] !=
                                   candidate_predictions[key]["predicted_idx"]]
                        grid.append({
                            "model": model_name, "tau": threshold,
                            "min_sequences": min_count,
                            "max_sequences": max_count,
                            "characters_compared_to_default": len(shared_chars),
                            "decision_flips_vs_default": len(changed),
                            "flips_corrected_vs_default": sum(
                                not default_predictions[model_name][key]["stopped_correct"] and
                                candidate_predictions[key]["stopped_correct"] for key in changed
                            ),
                            "flips_regressed_vs_default": sum(
                                default_predictions[model_name][key]["stopped_correct"] and
                                not candidate_predictions[key]["stopped_correct"] for key in changed
                            ),
                            **{k: summary[k] for k in (
                                "stopped_accuracy", "mean_sequences",
                                "itr_bits_per_minute", "sequence_row_column_nll",
                            )},
                        })
        output["policy_sweep"] = grid
        output["policy_selection"] = {
            "criterion": "fewest mean sequences at no pooled fused stopped-accuracy loss versus default; ties prefer higher accuracy then ITR",
            "best_per_model": {},
            "policy_models": policy_models,
        }
        for model_name in policy_models:
            candidates = [row for row in grid if row["model"] == model_name]
            default_accuracy = output["models"][model_name]["fusion"]["stopped_accuracy"]
            eligible = [row for row in candidates
                        if row["stopped_accuracy"] >= default_accuracy]
            if eligible:
                selected = min(eligible, key=lambda row: (
                    row["mean_sequences"], -row["stopped_accuracy"],
                    -row["itr_bits_per_minute"],
                ))
                selected = dict(selected)
                selected["default_accuracy_reference"] = default_accuracy
                selected["default_accuracy_difference"] = selected["stopped_accuracy"] - default_accuracy
            else:
                selected = {"model": model_name, "tau": tau,
                            "min_sequences": min_seq, "max_sequences": max_seq,
                            "stopped_accuracy": default_accuracy,
                            "mean_sequences": output["models"][model_name]["fusion"]["mean_sequences"],
                            "decision_flips_vs_default": 0,
                            "selection_note": "no swept policy retained default accuracy"}
            output["policy_selection"]["best_per_model"][model_name] = selected
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--study", default="StudyD")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--min-seq", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-sweep", action="store_true")
    parser.add_argument("--policy-models", nargs="+", default=None)
    args = parser.parse_args()
    result = run(args.scores, args.cache, args.study, args.alpha, args.tau,
                 args.min_seq, args.max_seq, args.output, args.policy_sweep,
                 args.policy_models)
    print(json.dumps({name: model["fusion"] for name, model in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
