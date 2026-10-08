#!/usr/bin/env python3
"""One-shot, lock-driven Study D manifest replay for clean M0 and M3a.

No model, calibration, fusion, or policy parameter is selected here. The only
held-out reads occur after the explicit evaluation opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

import h5py
import joblib
import numpy as np
from scipy.special import logsumexp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.spatialfilters import Xdawn
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_pipeline import load_spelling_matrix  # noqa: E402
from scripts.nested_sequence_calibration import _load_metadata, _predict  # noqa: E402
from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from scripts.run_riemann_cv import _load_tensor  # noqa: E402
from scripts.screen_shrinkage_lda import _decode_strings  # noqa: E402
from scripts.screen_shrinkage_lda import _balanced_fit_indices  # noqa: E402
from src.data.batch_preprocess import parse_bigp3bci_edf  # noqa: E402
from src.evaluation.sequence_scoring import (  # noqa: E402
    SequentialDecoder, aggregate_flash_logits, variable_time_itr,
)
from src.models.fusion import BayesianFusionEngine  # noqa: E402
from src.models.llm_predictor import LLMPredictor  # noqa: E402
from src.evaluation.classifier_protocol import load_heldout_run_ids  # noqa: E402
from src.preprocessing.build_session_sequence import yield_character_trials  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _condition(run_id):
    if "DynBigram" in run_id:
        return "DynBigram"
    if "Dyn" in run_id:
        return "Dyn"
    return "RC/Train"


def _channel_order(epochs, channels):
    names = list(epochs.ch_names)
    missing = [name for name in channels if name not in names]
    if missing:
        raise ValueError(f"recording lacks locked EEG channels: {missing}")
    epochs = epochs.copy().pick(channels)
    return epochs


def _features_m0(X, cfg, channels, sfreq, tmin):
    from scripts.run_classical_cv import _load_transform_batch
    return _load_transform_batch(X, cfg, channels, sfreq, tmin)


def _fit_final_models(cache_path, config_path, lock):
    cfg = json.loads(config_path.read_text())
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        meta["run_id"] = _decode_strings(meta["run_id"])
        y = meta["y"]
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        pool = np.arange(len(y))
        m0, scaler, _, _, _ = _fit_m0_streaming(
            h5["X"], pool, pool, y, cfg, channels, sfreq, tmin,
            int(lock["models"]["M0"].get("fit_sample_seed", 2000)),
        )
        fit_idx = _balanced_fit_indices(
            pool, y, int(lock["models"]["M3a"]["max_fit_samples"]),
            seed=int(lock["models"]["M3a"]["fit_sample_seed"]),
        )
        X_train, fit_idx = _load_tensor(
            h5["X"], fit_idx, cfg, channels, sfreq, tmin,
        )
        xdawn = Xdawn(nfilter=4, estimator="scm").fit(X_train, y[fit_idx])
        Z_train = xdawn.transform(X_train).reshape(len(X_train), -1)
        m3a = make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
            )
        ).fit(Z_train, y[fit_idx])
    models = {
        "M0": {"classifier": m0, "scaler": scaler,
               "temperature": lock["models"]["M0"]["sequence_temperature"]},
        "M3a": {"classifier": m3a, "xdawn": xdawn,
                "temperature": lock["models"]["M3a"]["sequence_temperature"]},
    }
    return models, channels, cfg, sfreq, tmin


def _score_epochs(models, X, channels, cfg, sfreq, tmin):
    scores = {}
    for name, model in models.items():
        if name == "M0":
            feats, keep = _features_m0(X, cfg, channels, sfreq, tmin)
            raw = np.full(len(X), np.nan)
            raw[keep] = model["classifier"].decision_function(
                model["scaler"].transform(feats[keep])
            )
        elif name == "M3a":
            times = tmin + np.arange(X.shape[-1]) / sfreq
            win = (times >= 0) & (times <= float(cfg["window_s"]) + 1e-8)
            tensor = X[:, :, win][:, :, ::int(cfg["decim"])]
            Z = model["xdawn"].transform(tensor).reshape(len(tensor), -1)
            raw = model["classifier"].decision_function(Z)
        else:
            raw = model["classifier"].decision_function(X)
        scores[name] = raw / float(model["temperature"])
    return scores


def _sequence_rows(logits, y, codes, chars, seqs, char_list, run_ids=None):
    by_char = defaultdict(list)
    seq_nll = defaultdict(list)
    run_ids = None if run_ids is None else np.asarray(run_ids).astype(str)
    group_values = (
        zip(run_ids.tolist(), chars.tolist(), seqs.tolist())
        if run_ids is not None else zip(chars.tolist(), seqs.tolist())
    )
    for group in sorted(set(group_values)):
        if run_ids is None:
            char_idx, seq_idx = group
            run_id = None
        else:
            run_id, char_idx, seq_idx = group
        # OOF cache uses zero-based sequence indices; parsed EDF metadata uses
        # one-based sequence_in_char. Both are valid as long as the index is nonnegative.
        if char_idx < 0 or seq_idx < 0:
            continue
        group_mask = (chars == char_idx) & (seqs == seq_idx)
        if run_ids is not None:
            group_mask &= run_ids == run_id
        ix = np.flatnonzero(group_mask)
        ix = ix[np.isfinite(logits[ix])]
        if not len(ix):
            continue
        target_codes = codes[ix][y[ix] == 1]
        tr = target_codes[(target_codes >= 1) & (target_codes <= 9)]
        tc = target_codes[(target_codes >= 10) & (target_codes <= 17)]
        evidence = aggregate_flash_logits(logits[ix], codes[ix], n_rows=9, n_cols=8)
        if evidence is None or not len(tr) or not len(tc):
            continue
        true_r, true_c = int(np.bincount(tr - 1).argmax()), int(np.bincount(tc - 10).argmax())
        rp, cp = evidence.row, evidence.column
        seq_nll[int(char_idx)].extend([
            -np.log(max(rp[true_r], 1e-15)),
            -np.log(max(cp[true_c], 1e-15)),
        ])
        key = (run_id, int(char_idx)) if run_ids is not None else int(char_idx)
        by_char[key].append((int(seq_idx), rp, cp, evidence.grid))
    return by_char, dict(seq_nll)


def _decode_oof_sequences(by_char, target_by_char, llm_priors, char_list,
                          alpha=0.0, tau=0.80, min_seq=2, max_seq=10):
    """Decode precomputed OOF row/column evidence without reading a manifest."""
    engine = BayesianFusionEngine(len(char_list), mode="fixed", base_alpha=alpha)
    decoded = []
    for (run_id, char_idx), sequences in sorted(by_char.items()):
        if (run_id, char_idx) not in target_by_char:
            continue
        target_idx = int(target_by_char[(run_id, char_idx)])
        initial = engine.get_initial_log_bias(llm_priors[(run_id, char_idx)]) if alpha else np.zeros(len(char_list))
        decoder = SequentialDecoder(len(char_list), initial, tau, min_seq, max_seq)
        ordered = sorted(sequences, key=lambda item: item[0])[:max_seq]
        full_log = initial.copy()
        for _, _, _, grid in ordered:
            full_log += np.log(np.clip(grid, 1e-9, 1.0))
        for _, _, _, grid in ordered:
            post, stopped = decoder.add(grid)
            if stopped:
                break
        result = decoder.result()
        if result["decision_idx"] is None:
            continue
        decoded.append({
            "subject": run_id.split("_SE", 1)[0], "run_id": run_id,
            "char_idx": int(char_idx), "target_idx": target_idx,
            "predicted_idx": result["decision_idx"],
            "stopped_correct": int(result["decision_idx"] == target_idx),
            "full_correct": int(int(full_log.argmax()) == target_idx),
            "sequences_used": result["sequences_used"],
        })
    return decoded


def _evaluate_run(run_id, raw_path, condition, lock, models, channels, cfg,
                  sfreq, tmin, spelling, char_list, llm, target_text):
    epochs, parsed_targets, _ = parse_bigp3bci_edf(
        raw_path, tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
        baseline_correction=False, decim=1, sequences_per_selection=20,
        verbose=False,
    )
    epochs = _channel_order(epochs, channels)
    if len(parsed_targets) != len(target_text):
        raise RuntimeError(
            f"{run_id}: EDF segmentation length differs from sealed target string "
            f"({len(parsed_targets)} vs {len(target_text)})"
        )
    X = epochs.get_data(copy=False).astype(np.float32, copy=False)
    y = epochs.events[:, 2].astype(np.uint8)
    md = epochs.metadata
    codes = md["stimulus_code"].to_numpy(dtype=np.int32)
    if {"char_index", "sequence_in_char"}.issubset(md.columns):
        chars = md["char_index"].to_numpy(dtype=np.int32)
        seqs = md["sequence_in_char"].to_numpy(dtype=np.int32)
    else:
        per_char = 17 * 10
        chars = np.arange(len(X), dtype=np.int32) // per_char
        seqs = ((np.arange(len(X), dtype=np.int32) // 17) % 10) + 1
    model_scores = _score_epochs(models, X, channels, cfg, sfreq, tmin)
    alpha = float(lock["fusion"]["alpha"])
    engine = BayesianFusionEngine(len(char_list), mode="fixed", base_alpha=alpha)
    sequence_records = {}
    sequence_nll_by_model = {}
    for name, logits in model_scores.items():
        sequence_records[name], sequence_nll_by_model[name] = _sequence_rows(
            logits, y, codes, chars, seqs, char_list
        )
    character_records = []
    n_chars = len(target_text)
    for char_idx in range(n_chars):
        target = target_text[char_idx]
        if target not in char_list:
            for name in models:
                character_records.append({
                    "run_id": run_id, "subject": run_id.split("_SE", 1)[0],
                    "condition": condition, "model": name, "char_idx": char_idx,
                    "target": target, "zero_coverage": True,
                    "decision_eeg": None, "decision_fused": None,
                    "correct_eeg": None, "correct_fused": None,
                    "flashes_used": 0, "eeg_flashes_used": 0,
                    "posterior_at_stop": None, "eeg_posterior_at_stop": None,
                })
            continue
        target_idx = char_list.index(target)
        context = target_text[:char_idx]
        prior = llm.predict_next_char(context)
        initial = engine.get_initial_log_bias(prior)
        model_state = {}
        for name in models:
            policy_name = "M0" if name == "OldSWLDA_contaminated" else name
            policy = lock["stopping_policy"][policy_name]
            model_state[name] = {
                "eeg_decoder": SequentialDecoder(
                    len(char_list), tau=policy["tau"],
                    min_sequences=policy["min_sequences"],
                    max_sequences=policy["max_sequences"],
                ),
                "fused_decoder": SequentialDecoder(
                    len(char_list), initial, tau=policy["tau"],
                    min_sequences=policy["min_sequences"],
                    max_sequences=policy["max_sequences"],
                ),
                "fused_stopped": False,
                "eeg_stopped": False,
            }
        sequences = sorted(set(
            seq for cidx, seq in zip(chars, seqs) if int(cidx) == char_idx and int(seq) > 0
        ))
        valid_count = 0
        for sequence in sequences[:max(
            lock["stopping_policy"]["M0" if name == "OldSWLDA_contaminated" else name]["max_sequences"] for name in models
        )]:
            # Construct row/column posterior from classifier per-flash logits.
            ix = np.flatnonzero((chars == char_idx) & (seqs == sequence))
            if not len(ix):
                continue
            usable_sequence = False
            for name in models:
                policy = lock["stopping_policy"]["M0" if name == "OldSWLDA_contaminated" else name]
                state = model_state[name]
                if state["fused_stopped"] and state["eeg_stopped"]:
                    continue
                logits = model_scores[name][ix]
                evidence = aggregate_flash_logits(logits, codes[ix], n_rows=9, n_cols=8)
                if evidence is None:
                    continue
                usable_sequence = True
                if not state["eeg_stopped"]:
                    eeg_post, eeg_stopped = state["eeg_decoder"].add(evidence.grid)
                    if eeg_stopped:
                        state["eeg_stopped"] = True
                        state["decision_eeg"] = int(eeg_post.argmax())
                        state["eeg_posterior_at_stop"] = float(eeg_post.max())
                if not state["fused_stopped"]:
                    fused_post, fused_stopped = state["fused_decoder"].add(evidence.grid)
                    if fused_stopped:
                        state["fused_stopped"] = True
                        state["decision_fused"] = int(fused_post.argmax())
                        state["posterior_at_stop"] = float(fused_post.max())
            if not usable_sequence:
                continue
            valid_count += 1
            for state in model_state.values():
                if state["eeg_stopped"]:
                    state["eeg_sequences_used"] = state["eeg_decoder"].sequences_used
                if state["fused_stopped"]:
                    state["sequences_used"] = state["fused_decoder"].sequences_used
            if all(s["fused_stopped"] and s["eeg_stopped"] for s in model_state.values()):
                break
        for name, state in model_state.items():
            policy_name = "M0" if name == "OldSWLDA_contaminated" else name
            fused_result = state["fused_decoder"].result()
            eeg_result = state["eeg_decoder"].result()
            used = fused_result["sequences_used"]
            eeg_used = eeg_result["sequences_used"]
            if not state["fused_stopped"]:
                state["decision_fused"] = fused_result["decision_idx"]
                state["posterior_at_stop"] = fused_result["confidence"]
            if not state["eeg_stopped"]:
                state["decision_eeg"] = eeg_result["decision_idx"]
                state["eeg_posterior_at_stop"] = eeg_result["confidence"]
            character_records.append({
                "run_id": run_id, "subject": run_id.split("_SE", 1)[0],
                "condition": condition, "model": name, "char_idx": char_idx,
                "target": target,
                "decision_eeg": char_list[state["decision_eeg"]],
                "decision_fused": char_list[state["decision_fused"]],
                "correct_eeg": state["decision_eeg"] == target_idx,
                "correct_fused": state["decision_fused"] == target_idx,
                "flashes_used": int(used * 17),
                "sequences_used": int(used),
                "eeg_flashes_used": int(eeg_used * 17),
                "eeg_sequences_used": int(eeg_used),
                "sequence_nll": float(np.mean(sequence_nll_by_model[name].get(char_idx, [])))
                    if sequence_nll_by_model[name].get(char_idx) else None,
                "posterior_at_stop": state["posterior_at_stop"],
                "eeg_posterior_at_stop": state["eeg_posterior_at_stop"],
                "zero_coverage": valid_count == 0,
            })
    return character_records, sequence_nll_by_model


def _aggregate(records, model, condition=None):
    rows = [r for r in records if r.get("model") == model and
            (condition is None or (condition == "Dyn/DynBigram" and r["condition"] in {"Dyn", "DynBigram"}) or r["condition"] == condition)]
    usable = [r for r in rows if not r.get("zero_coverage")]
    n = len(usable)
    if not n:
        return {"characters": 0, "zero_coverage_terminal_characters": len(rows)}
    eeg_acc = float(np.mean([r["correct_eeg"] for r in usable]))
    fused_acc = float(np.mean([r["correct_fused"] for r in usable]))
    seqs = float(np.mean([r["sequences_used"] for r in usable]))
    mean_seconds = float(np.mean([r["flashes_used"] * 2.0 / 17.0 for r in usable]))
    itr = variable_time_itr(72, fused_acc, mean_seconds)
    by_subject = {}
    for subject in sorted({r["subject"] for r in usable}):
        sr = [r for r in usable if r["subject"] == subject]
        by_subject[subject] = {
            "characters": len(sr),
            "eeg_accuracy": float(np.mean([r["correct_eeg"] for r in sr])),
            "fused_accuracy": float(np.mean([r["correct_fused"] for r in sr])),
            "sequence_row_column_nll": float(np.mean([
                r["sequence_nll"] for r in sr if r.get("sequence_nll") is not None
            ])) if any(r.get("sequence_nll") is not None for r in sr) else None,
        }
    return {
        "characters": n, "zero_coverage_terminal_characters": len(rows) - n,
        "eeg_accuracy": eeg_acc, "fused_accuracy": fused_acc,
        "mean_sequences": seqs,
        "eeg_flashes_per_character": float(np.mean([r["eeg_flashes_used"] for r in usable])),
        "flashes_per_character": float(np.mean([r["flashes_used"] for r in usable])),
        "itr_bits_per_minute": itr, "wpm_approx": itr / 5.0,
        "sequence_row_column_nll": float(np.mean([r["sequence_nll"] for r in usable if r.get("sequence_nll") is not None])) if any(r.get("sequence_nll") is not None for r in usable) else None,
        "per_subject": by_subject,
    }


def _paired_subject_statistics(records, baseline="M0", candidates=("M3a",), seed=42):
    rng = np.random.default_rng(seed)
    comparisons = []
    base = _aggregate(records, baseline, None)["per_subject"]
    for model in candidates:
        candidate = _aggregate(records, model, None)["per_subject"]
        subjects = sorted(set(base) & set(candidate))
        if not subjects:
            continue
        metric_specs = (
            ("fused character accuracy", "fused_accuracy", True),
            ("sequence row/column NLL", "sequence_row_column_nll", False),
        )
        for label, key, primary in metric_specs:
            available = [s for s in subjects if base[s].get(key) is not None and candidate[s].get(key) is not None]
            deltas = np.asarray([candidate[s][key] - base[s][key] for s in available], dtype=float)
            try:
                statistic, p_value = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            boot = np.mean(rng.choice(deltas, size=(20000, len(deltas)), replace=True), axis=1)
            comparisons.append({
                "comparison": f"{model} vs {baseline}", "metric": label,
                "holm_family_member": primary,
                "subjects": len(available), "subject_ids": available,
                "mean_paired_delta": float(deltas.mean()),
                "subjects_better": int(np.sum(deltas > 0)) if primary else None,
                "subjects_tied": int(np.sum(deltas == 0)) if primary else None,
                "subjects_worse": int(np.sum(deltas < 0)) if primary else None,
                "wilcoxon_statistic": float(statistic),
                "p_uncorrected": float(p_value),
                "bootstrap_95_ci_subject_resampling": [
                    float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
                ],
            })
    family_idxs = [i for i, row in enumerate(comparisons) if row["holm_family_member"]]
    order = sorted(family_idxs, key=lambda i: comparisons[i]["p_uncorrected"])
    adjusted = np.full(len(comparisons), np.nan, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(order) - rank) * comparisons[idx]["p_uncorrected"])
        adjusted[idx] = min(1.0, running)
    for row, p_adj in zip(comparisons, adjusted):
        row["p_holm"] = float(p_adj) if np.isfinite(p_adj) else None
        row["significant_holm_0p05"] = bool(p_adj < .05) if np.isfinite(p_adj) else None
    return {"correction": "Holm across finalist vs M0 fused-accuracy comparisons; maximum two. NLL is secondary and reported separately.",
            "comparisons": comparisons}


def run(args):
    lock_path, manifest_path = args.lock, args.manifest
    lock = json.loads(lock_path.read_text())
    if _sha(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("manifest SHA-256 differs from finalist lock")
    if not args.allow_heldout_eval and not args.dry_run_training:
        raise RuntimeError("Pass --allow-heldout-eval for the one-shot manifest replay")
    lock_hash = _sha(lock_path)
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    models, channels, cfg, sfreq, tmin = _fit_final_models(args.cache, args.config, lock)
    if not args.dry_run_training:
        models["OldSWLDA_contaminated"] = {
            "classifier": joblib.load(args.old_model),
            "temperature": 1.0,
            "audit_only": True,
        }
    spelling, _, _ = load_spelling_matrix("data/processed/grid_layout.json", study_name="StudyD")
    char_list = list(np.asarray(spelling).ravel())
    llm = LLMPredictor(spelling, local_files_only=True)
    target_trials = yield_character_trials(
        "data/processed/ground_truth_registry.json", None, study="StudyD",
        include_heldout=bool(args.allow_heldout_eval and not args.dry_run_training),
    )
    targets_by_run = defaultdict(list)
    for trial in target_trials:
        targets_by_run[trial["session_id"]].append(trial["target_char"])
    edf_paths = {p.stem: p for p in args.raw_dir.rglob("*.edf") if not p.name.startswith("._")}
    if args.dry_run_training:
        ids = [args.dry_run_training]
    else:
        manifest = json.loads(manifest_path.read_text())
        ids = [rid for values in manifest.values() for rid in values]
        if len(ids) != 57 or len(set(ids)) != 57:
            raise RuntimeError("manifest must contain exactly 57 unique runs")
    missing = [rid for rid in ids if rid not in edf_paths]
    if missing:
        raise FileNotFoundError(f"raw EDFs missing for {len(missing)} selected runs")
    missing_targets = [rid for rid in ids if rid not in targets_by_run]
    if missing_targets:
        raise RuntimeError(f"label vault has no targets for {len(missing_targets)} selected runs")
    all_records = []
    for i, run_id in enumerate(ids, start=1):
        records, _ = _evaluate_run(
            run_id, edf_paths[run_id], _condition(run_id), lock, models,
            channels, cfg, sfreq, tmin, spelling, char_list, llm,
            "".join(targets_by_run[run_id]),
        )
        all_records.extend(records)
        print(f"processed {i}/{len(ids)}: {run_id}", flush=True)
    summaries = {}
    for model in models:
        summaries[model] = {
            condition: _aggregate(all_records, model, condition)
            for condition in (None, "RC/Train", "Dyn/DynBigram")
        }
    output = {
        "status": "training_pool_smoke" if args.dry_run_training else "completed_one_shot_manifest_eval",
        "lock_sha256": lock_hash, "manifest_sha256": lock["manifest_sha256"],
        "git_hash": git_hash, "lock_path": str(lock_path),
        "manifest_run_count": len(ids), "runs_processed": ids,
        "records": all_records, "summaries": summaries,
        "statistics": _paired_subject_statistics(all_records),
        "audit_comparison": {
            "model": "OldSWLDA_contaminated",
            "reference": "M0",
            "policy": "clean M0 locked policy applied unchanged to both models",
            "role": "descriptive audit only; excluded from finalist selection and Holm family",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--allow-heldout-eval", action="store_true")
    p.add_argument("--dry-run-training", help="Run one named training run only; never accesses manifest runs")
    p.add_argument("--lock", type=Path, default=ROOT / "splits/finalist_lock.json")
    p.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    p.add_argument("--cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline.h5")
    p.add_argument("--config", type=Path, default=ROOT / "configs/classifier_optimization/preprocessing/baseline_off.json")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bigP3BCI_dataset")
    p.add_argument("--old-model", type=Path, default=ROOT / "data/processed/swlda_model.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_manifest_eval.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
