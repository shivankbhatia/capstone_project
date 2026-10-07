#!/usr/bin/env python3
"""One-shot, lock-verified five-rung Study D replay using clean M0."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import softmax
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import load_spelling_matrix  # noqa: E402
from scripts.evaluate_classifier_manifest import _channel_order  # noqa: E402
from src.data.batch_preprocess import parse_bigp3bci_edf  # noqa: E402
from src.models.decoder import calculate_itr  # noqa: E402
from src.models.fusion import BayesianFusionEngine  # noqa: E402
from src.models.llm_predictor import LLMPredictor  # noqa: E402
from src.models.rag_predictor import RAGPredictor  # noqa: E402


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _condition(run_id):
    if "_DynBigram_" in run_id:
        return "DynBigram"
    if "_Dyn_" in run_id:
        return "Dyn"
    return "RC/Train"


def _split_target_keys(target_text, char_list):
    """Recover grid-key boundaries from concatenated channel labels.

    Multi-character key labels (for example ``Pause`` and ``PgUp``) are
    concatenated without delimiters by the EDF parser, while ordinary text
    labels are one character each. Prefer the longest exact grid label and
    uppercase single-letter targets as a fallback for lowercase Dyn strings.
    """
    keys = sorted((str(key) for key in char_list), key=lambda key: (-len(key), key))
    result = []
    offset = 0
    while offset < len(target_text):
        exact = next((key for key in keys if target_text.startswith(key, offset)), None)
        if exact is not None:
            result.append(exact)
            offset += len(exact)
            continue
        upper = target_text[offset].upper()
        if len(target_text[offset]) == 1 and upper in char_list:
            result.append(upper)
            offset += 1
            continue
        raise RuntimeError(f"Could not map target text at offset {offset}: {target_text[offset:offset + 12]!r}")
    return result


def _context_for_keys(keys, llm):
    chunks = []
    for key in keys:
        text = llm._candidate_text(key)
        if len(key) > 1 and len(text.strip()) > 1:
            chunks.append(" " + text.strip() + " ")
        else:
            chunks.append(text)
    return "".join(chunks).strip()


def _softmax_log_prior(prior, rung, lock):
    rung_cfg = lock["rungs"][str(rung)]
    if rung == 0:
        return np.zeros(len(prior), dtype=float)
    alpha = float(rung_cfg["fusion_weight"] if rung == 1 else rung_cfg["adaptive_weight"])
    mode = "fixed" if rung == 1 else "adaptive"
    engine = BayesianFusionEngine(len(prior), mode=mode, base_alpha=alpha)
    return engine.get_initial_log_bias(prior)


def _make_predictors(base, subject, lock):
    global_cfg = lock["rungs"]["2"]
    subject_cfg = lock["rungs"]["3"]
    personal_cfg = lock["rungs"]["4"]
    common = dict(
        phrase_bank_path=lock["rungs"]["2"]["rag_bank"],
        retrieval_confidence_threshold=global_cfg["retrieval_confidence_threshold"],
        min_subject_phrases=lock["rag"]["backoff_gate"]["subject_min_unique_phrases"],
        personalization_bonus=lock["rag"]["backoff_gate"]["personalization_bonus"],
        subject_confidence_threshold=lock["rag"]["backoff_gate"]["subject_confidence_threshold"],
        sufficiency_midpoint_tokens=lock["rag"]["sufficiency_gate"]["midpoint_tokens"],
        sufficiency_sharpness=lock["rag"]["sufficiency_gate"]["sharpness"],
        bigram_min_count=lock["rag"]["backoff_gate"]["min_successor_count"],
        bigram_max_normalized_entropy=lock["rag"]["backoff_gate"]["max_normalized_entropy"],
    )
    return {
        2: RAGPredictor(base, rag_weight=global_cfg["rag_weight"], **common),
        3: RAGPredictor(base, subject_id=subject, subject_only=True,
                        rag_weight=subject_cfg["rag_weight"], **common),
        4: RAGPredictor(base, subject_id=subject,
                        rag_weight=personal_cfg["rag_weight"], **common),
    }


def _sequence_evidence(logits, codes):
    rows, cols = np.zeros(9), np.zeros(8)
    valid = []
    for logit, code in zip(logits, codes):
        if not np.isfinite(logit):
            continue
        code = int(code)
        if 1 <= code <= 9:
            rows[code - 1] += float(logit)
            valid.append(code)
        elif 10 <= code <= 17:
            cols[code - 10] += float(logit)
            valid.append(code)
    if not any(1 <= code <= 9 for code in valid) or not any(10 <= code <= 17 for code in valid):
        return None
    row_probs, col_probs = softmax(rows), softmax(cols)
    return {
        "grid": np.outer(row_probs, col_probs).ravel(),
        "row": row_probs,
        "column": col_probs,
        "valid_flash_count": len(valid),
    }


def _evaluate_run(run_id, edf_path, target_text, lock, scorer, spelling, char_list, llm):
    epochs, parsed_targets, _ = parse_bigp3bci_edf(
        str(edf_path), tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
        baseline_correction=False, decim=1, sequences_per_selection=20,
        verbose=False,
    )
    target_keys = _split_target_keys(target_text, char_list)
    parsed_keys = _split_target_keys(parsed_targets, char_list)
    if len(parsed_keys) != len(target_keys):
        raise RuntimeError(f"{run_id}: parsed target length differs from vault labels")
    channels = list(scorer.channels)
    epochs = _channel_order(epochs, channels)
    X = epochs.get_data(copy=False).astype(np.float32, copy=False)
    logits = np.asarray(scorer.calibrated_logits(X), dtype=float)
    md = epochs.metadata
    codes = md["stimulus_code"].to_numpy(dtype=np.int32)
    if md is not None and {"char_index", "sequence_in_char"}.issubset(md.columns):
        chars = md["char_index"].to_numpy(dtype=np.int32)
        seqs = md["sequence_in_char"].to_numpy(dtype=np.int32)
    else:
        # Fixed RC acquisition depth is ten sequences per six-character block.
        per_char = 17 * int(lock["stopping"]["rc_sequences_per_char"])
        chars = np.arange(len(X), dtype=np.int32) // per_char
        seqs = ((np.arange(len(X), dtype=np.int32) // 17) % 10) + 1

    run_id = str(run_id)
    subject = run_id.split("_SE", 1)[0]
    predictors = _make_predictors(llm, subject, lock)
    all_records = []
    max_sequences = int(lock["stopping"]["max_sequences"])
    min_sequences = int(lock["stopping"]["min_sequences"])
    tau = float(lock["stopping"]["tau"])
    seq_seconds = 2.0
    seconds_per_flash = seq_seconds / 17.0

    for char_idx, target in enumerate(target_keys):
        target_idx = char_list.index(target)
        context = _context_for_keys(target_keys[:char_idx], llm)
        base_prior = llm.predict_next_char(context)
        priors = {1: base_prior}
        rag_diag = {}
        for rung in (2, 3, 4):
            predictor = predictors[rung]
            priors[rung] = predictor.predict_next_char(context, target_char=target)
            diag = predictor.last_diagnostics
            rag_diag[rung] = {
                "subject_gate_weight": float(diag.subject_gate_weight),
                "global_gate_weight": float(diag.global_gate_weight),
                "sufficiency_gate_value": float(diag.sufficiency_gate_value),
                "subject_source": diag.subject_source,
            }

        states = {}
        for rung in range(5):
            initial = _softmax_log_prior(priors.get(rung, base_prior), rung, lock)
            states[rung] = {
                "accumulated": initial.copy(), "decision_idx": None,
                "confidence": None, "target_posterior": None,
                "sequences_used": 0, "flashes_used": 0,
                "sequence_nll_values": [],
            }

        sequences = sorted({int(s) for c, s in zip(chars, seqs)
                            if int(c) == char_idx and int(s) > 0})
        valid_count = 0
        for sequence in sequences:
            if valid_count >= max_sequences:
                break
            ix = np.flatnonzero((chars == char_idx) & (seqs == sequence))
            evidence = _sequence_evidence(logits[ix], codes[ix])
            if evidence is None:
                continue
            valid_count += 1
            for rung, state in states.items():
                if state["decision_idx"] is not None:
                    continue
                state["accumulated"] += np.log(np.clip(evidence["grid"], 1e-12, 1.0))
                state["sequences_used"] = valid_count
                state["flashes_used"] += evidence["valid_flash_count"]
                target_row, target_col = divmod(target_idx, 8)
                state["sequence_nll_values"].append(float(
                    -np.log(max(evidence["row"][target_row], 1e-15))
                    -np.log(max(evidence["column"][target_col], 1e-15))
                ))
                posterior = softmax(state["accumulated"])
                if valid_count >= min_sequences and posterior.max() >= tau:
                    state["decision_idx"] = int(posterior.argmax())
                    state["confidence"] = float(posterior.max())
                    state["target_posterior"] = float(posterior[target_idx])
            # Each rung stops independently; future sequences are needed only
            # for rungs whose posterior has not crossed the locked threshold.
            if all(state["decision_idx"] is not None for state in states.values()):
                break

        for rung, state in states.items():
            if valid_count == 0:
                all_records.append({
                    "run_id": run_id, "subject": subject, "condition": _condition(run_id),
                    "rung": rung, "char_idx": char_idx, "target": target,
                    "decision": None, "correct": None, "zero_coverage": True,
                    "flashes_used": 0, "sequences_used": 0,
                    "posterior_at_stop": None, "target_posterior_at_stop": None,
                    "sequence_nll": None,
                })
                continue
            posterior = softmax(state["accumulated"])
            decision_idx = state["decision_idx"]
            if decision_idx is None:
                decision_idx = int(posterior.argmax())
            all_records.append({
                "run_id": run_id, "subject": subject, "condition": _condition(run_id),
                "rung": rung, "char_idx": char_idx, "target": target,
                "decision": char_list[decision_idx], "correct": decision_idx == target_idx,
                "zero_coverage": False,
                "flashes_used": int(state["flashes_used"]),
                "sequences_used": int(state["sequences_used"]),
                "posterior_at_stop": float(state["confidence"] or posterior.max()),
                "target_posterior_at_stop": float(state["target_posterior"] or posterior[target_idx]),
                "sequence_nll": float(np.mean(state["sequence_nll_values"]))
                    if state["sequence_nll_values"] else None,
                "time_seconds": float(state["flashes_used"] * seconds_per_flash),
                "rag_diagnostics": rag_diag.get(rung),
            })

    by_key = {(r["rung"], r["char_idx"]): r for r in all_records}
    for row in all_records:
        rung1 = by_key[(1, row["char_idx"])]
        row["argmax_flip_flag_vs_rung1"] = (
            row["decision"] != rung1["decision"]
            if row["decision"] is not None and rung1["decision"] is not None
            else None
        )
        row["flip_toward_target_vs_rung1"] = bool(
            row["argmax_flip_flag_vs_rung1"] and row["correct"] and not rung1["correct"]
        )
        row["flip_away_from_target_vs_rung1"] = bool(
            row["argmax_flip_flag_vs_rung1"] and not row["correct"] and rung1["correct"]
        )
        if row["rung"] == 4 and row["rag_diagnostics"]:
            row["sufficiency_gate_deferred_to_global"] = (
                row["rag_diagnostics"]["subject_gate_weight"] <= 1e-12
            )
    return all_records


def _aggregate(records, rung, condition=None):
    rows = [row for row in records if row["rung"] == rung and
            (condition is None or (condition == "Dyn/DynBigram" and
             row["condition"] in {"Dyn", "DynBigram"}) or row["condition"] == condition)]
    usable = [row for row in rows if not row["zero_coverage"]]
    if not usable:
        return {"characters": 0, "zero_coverage_terminal_characters": len(rows)}
    accuracy = float(np.mean([row["correct"] for row in usable]))
    mean_seconds = float(np.mean([row["time_seconds"] for row in usable]))
    itr = calculate_itr(72, accuracy * 100.0, mean_seconds)
    return {
        "characters": len(usable),
        "zero_coverage_terminal_characters": len(rows) - len(usable),
        "character_accuracy": accuracy,
        "flashes_per_character": float(np.mean([row["flashes_used"] for row in usable])),
        "sequences_per_character": float(np.mean([row["sequences_used"] for row in usable])),
        "mean_time_seconds": mean_seconds,
        "itr_variable_time_bits_per_minute": itr,
        "wpm": itr / 5.0,
        "sequence_nll": float(np.mean([row["sequence_nll"] for row in usable
                                        if row["sequence_nll"] is not None])),
        "per_subject": _subject_aggregates(usable),
        "flip_diagnostics_vs_rung1": {
            "flips": sum(bool(row.get("argmax_flip_flag_vs_rung1")) for row in usable),
            "toward_target": sum(bool(row.get("flip_toward_target_vs_rung1")) for row in usable),
            "away_from_target": sum(bool(row.get("flip_away_from_target_vs_rung1")) for row in usable),
            "sufficiency_gate_deferred_to_global_share": (
                float(np.mean([row.get("sufficiency_gate_deferred_to_global", False)
                               for row in usable])) if rung == 4 else None
            ),
        },
    }


def _subject_aggregates(rows):
    result = {}
    for subject in sorted({row["subject"] for row in rows}):
        data = [row for row in rows if row["subject"] == subject]
        result[subject] = {
            "characters": len(data),
            "accuracy": float(np.mean([row["correct"] for row in data])),
            "flashes_per_character": float(np.mean([row["flashes_used"] for row in data])),
            "sequence_nll": float(np.mean([row["sequence_nll"] for row in data
                                            if row["sequence_nll"] is not None])),
        }
    return result


def _statistics(records, seed=0, n_boot=10000):
    comparisons = ((1, 0), (2, 1), (3, 1), (4, 1), (4, 2))
    summaries = {rung: _aggregate(records, rung)["per_subject"] for rung in range(5)}
    raw = []
    rng = np.random.default_rng(seed)
    for candidate, baseline in comparisons:
        subjects = sorted(set(summaries[candidate]) & set(summaries[baseline]))
        differences = np.asarray([
            summaries[candidate][s]["accuracy"] - summaries[baseline][s]["accuracy"]
            for s in subjects
        ], dtype=float)
        if not len(differences):
            continue
        p_value = float(wilcoxon(differences).pvalue) if np.any(differences) else 1.0
        samples = rng.choice(differences, size=(n_boot, len(differences)), replace=True).mean(axis=1)
        raw.append({
            "comparison": f"{candidate}v{baseline}", "candidate_rung": candidate,
            "baseline_rung": baseline, "subjects": subjects,
            "n_subjects": len(subjects), "mean_accuracy_difference": float(differences.mean()),
            "median_accuracy_difference": float(np.median(differences)),
            "subjects_better": int(np.sum(differences > 0)),
            "subjects_tied": int(np.sum(differences == 0)),
            "subjects_worse": int(np.sum(differences < 0)),
            "wilcoxon_p": p_value,
            "bootstrap_subject_ci_95": [float(np.quantile(samples, .025)),
                                        float(np.quantile(samples, .975))],
        })
    ordered = sorted(range(len(raw)), key=lambda i: raw[i]["wilcoxon_p"])
    adjusted = [None] * len(raw)
    running = 0.0
    for rank, idx in enumerate(ordered):
        running = max(running, min(1.0, (len(raw) - rank) * raw[idx]["wilcoxon_p"]))
        adjusted[idx] = running
    for row, p_adj in zip(raw, adjusted):
        row["holm_p"] = p_adj
        row["holm_significant_0p05"] = p_adj < .05
    return {"primary_metric": "per_subject_fused_character_accuracy",
            "family": [f"{a}v{b}" for a, b in comparisons],
            "correction": "Holm", "bootstrap": {"unit": "subject", "n_boot": n_boot,
                                                     "seed": seed},
            "comparisons": raw}


def run(lock_path, manifest_path, cache_path, raw_dir, result_path, allow_heldout_eval):
    if not allow_heldout_eval:
        raise RuntimeError("Explicit --allow-heldout-eval is required after the ablation lock is committed")
    if result_path.exists():
        raise FileExistsError(f"One-shot result already exists: {result_path}")
    if _sha(manifest_path) != json.loads(lock_path.read_text())["parent_locks"]["manifest_sha256"]:
        raise RuntimeError("Held-out manifest hash differs from the committed ablation lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    classifier_lock = ROOT / "splits/finalist_lock.json"
    if _sha(classifier_lock) != lock["parent_locks"]["classifier_lock_sha256"]:
        raise RuntimeError("Classifier parent lock hash mismatch")
    if not (ROOT / lock["classifier"]["scorer_path"]).exists():
        raise FileNotFoundError("Export the locked clean M0 scorer before evaluation")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(lock_path.relative_to(ROOT))],
                             cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode:
        raise RuntimeError("ablation_lock.json must be committed before evaluation")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [run_id for runs in manifest.values() for run_id in runs]
    if len(ids) != 57 or len(set(ids)) != 57:
        raise RuntimeError("The frozen manifest must contain 57 unique runs")
    vault_path = ROOT / "data/evaluation/ground_truth_vault.json"
    targets = json.loads(vault_path.read_text(encoding="utf-8"))
    if set(ids) - set(targets):
        raise RuntimeError("Evaluation vault is missing manifest targets")
    raw_paths = {path.stem: path for path in raw_dir.rglob("*.edf")
                 if not path.name.startswith("._")}
    if set(ids) - set(raw_paths):
        raise FileNotFoundError(f"Raw EDF missing for {len(set(ids) - set(raw_paths))} manifest runs")

    spelling, _, _ = load_spelling_matrix("data/processed/grid_layout.json", study_name="StudyD")
    char_list = list(np.asarray(spelling).ravel())
    llm = LLMPredictor(spelling, local_files_only=True,
                       special_key_map_path=ROOT / lock["lm"]["special_key_map_path"])
    scorer = joblib.load(ROOT / lock["classifier"]["scorer_path"])
    records = []
    for index, run_id in enumerate(ids, start=1):
        records.extend(_evaluate_run(
            run_id, raw_paths[run_id], targets[run_id], lock, scorer,
            spelling, char_list, llm,
        ))
        print(f"processed {index}/{len(ids)}: {run_id}", flush=True)

    summaries = {
        str(rung): {
            stratum: _aggregate(records, rung, condition)
            for stratum, condition in (("overall", None), ("RC/Train", "RC/Train"),
                                       ("Dyn/DynBigram", "Dyn/DynBigram"))
        }
        for rung in range(5)
    }
    output = {
        "status": "completed_one_shot_ablation_ladder",
        "git_hash": head,
        "ablation_lock_sha256": _sha(lock_path),
        "classifier_lock_sha256": lock["parent_locks"]["classifier_lock_sha256"],
        "manifest_sha256": lock["parent_locks"]["manifest_sha256"],
        "runs_processed": ids,
        "run_count": len(ids),
        "records": records,
        "summaries": summaries,
        "statistics": _statistics(records, seed=lock["evaluation"]["stats"]["seed"],
                                   n_boot=lock["evaluation"]["stats"]["n_boot"]),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-heldout-eval", action="store_true")
    parser.add_argument("--lock", type=Path, default=ROOT / "splits/ablation_lock.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline.h5")
    parser.add_argument("--raw-dir", type=Path,
                        default=ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyD")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/tables/ablation_ladder_clean_m0.json")
    args = parser.parse_args()
    result = run(args.lock, args.manifest, args.cache, args.raw_dir,
                 args.output, args.allow_heldout_eval)
    print(json.dumps({r: result["summaries"][r]["overall"] for r in result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
