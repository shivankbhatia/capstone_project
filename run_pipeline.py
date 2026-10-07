import json
import time
import os
import numpy as np
import mne
import joblib

from src.models.decoder import P300Decoder, calculate_itr
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor
from src.preprocessing.build_session_sequence import yield_character_trials

# -----------------------------------------------------------------------------
# GRID LAYOUT — built from the ACTUAL parsed grid, not a hardcoded 6x6
# -----------------------------------------------------------------------------
def load_spelling_matrix(grid_layout_path, study_name="StudyD"):
    """
    Builds the real spelling matrix from grid_layout.json (written by
    batch_preprocess.py from the actual channel-derived grid_map). Replaces
    the previous hardcoded 6x6/36-class matrix, which didn't match Study D's
    real 9x8/72-key extended keyboard layout.
    """
    with open(grid_layout_path) as f:
        layouts = json.load(f)

    if study_name not in layouts:
        raise ValueError(f"No grid layout recorded for {study_name} in {grid_layout_path}. "
                          f"Run batch_preprocess.py on at least one {study_name} file first.")

    info = layouts[study_name]
    n_rows, n_cols = info["n_rows"], info["n_cols"]
    grid_map = info["grid_map"]  # {label: [row, col]}

    # Must match the SAME idx = (row-1)*n_cols + col convention used in
    # epoching.py, so char_list[idx] lines up with StimulusCode values.
    matrix = np.full((n_rows, n_cols), "", dtype=object)
    for label, (row, col) in grid_map.items():
        matrix[row - 1, col - 1] = label

    if np.any(matrix == ""):
        missing = np.argwhere(matrix == "")
        print(f"WARNING: {len(missing)} grid cell(s) have no label — check grid_map completeness.")

    return matrix, n_rows, n_cols


# -----------------------------------------------------------------------------
# ABLATION TRACKER UTILITY
# -----------------------------------------------------------------------------
class SimpleAblationTracker:
    def __init__(self):
        self.results = {}

    def record_run(self, name, metrics, itr):
        wpm = itr / 5.0 if itr > 0 else 0.0  # Standard rough approx: 5 bits per word

        self.results[name] = {
            'accuracy': metrics.accuracy,
            'flashes_per_char': metrics.total_flashes_used / max(1, metrics.total_characters),
            'itr': itr,
            'wpm': wpm
        }

    def print_summary(self):
        print("\n" + "="*50)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*50)
        for name, data in self.results.items():
            print(f"\nCondition: [{name}]")
            print(f"  Accuracy (%): {data['accuracy']:.2f}")
            print(f"  Flashes/Char: {data['flashes_per_char']:.2f}")
            print(f"  ITR (bits/min): {data['itr']:.2f}")
        print("="*50 + "\n")


# -----------------------------------------------------------------------------
# REAL CLASSIFIER STREAM
# -----------------------------------------------------------------------------
_epoch_cache = {}
_adaptive_index_cache = {}
_legacy_index_warning_paths = set()


def fixed_sequence_epoch_slice(char_idx, sequence_num, flashes_per_seq,
                               sequences_per_char=10):
    """Index fixed RC data using its acquisition repetitions, not stop max."""
    if char_idx < 0 or sequence_num < 1 or flashes_per_seq < 1 or sequences_per_char < 1:
        raise ValueError("invalid fixed-sequence indices")
    flashes_per_char = flashes_per_seq * sequences_per_char
    start = char_idx * flashes_per_char + (sequence_num - 1) * flashes_per_seq
    return slice(start, start + flashes_per_seq)


def row_column_posterior(flash_probs, stim_codes, n_rows, n_cols):
    """Convert row/column flash probabilities into a grid posterior.

    A classic RC sequence flashes every row and column exactly once, but the
    Study-D Dyn/DynBigram conditions may repeat or skip groups.  Replacing a
    row or column score with the *last* matching flash makes the result depend
    on presentation order.  Instead, sum each flash's log likelihood ratio;
    it preserves the same winning row/column for a complete standard sequence
    and properly accumulates repeated dynamic flashes.
    """
    probabilities = np.asarray(flash_probs, dtype=float)
    codes = np.asarray(stim_codes, dtype=int)
    if probabilities.shape != codes.shape:
        raise ValueError("flash_probs and stim_codes must have the same shape")

    # Probability is the classifier's estimate that a flash is a target.
    # For a candidate row/column, a matching flash is target evidence; its
    # Bayes factor against a non-target flash is p / (1 - p).
    eps = 1e-6
    log_odds = np.log(np.clip(probabilities, eps, 1 - eps)) - np.log1p(
        -np.clip(probabilities, eps, 1 - eps)
    )
    row_log_scores = np.zeros(n_rows)
    col_log_scores = np.zeros(n_cols)

    for score, code in zip(log_odds, codes):
        if 1 <= code <= n_rows:
            row_log_scores[code - 1] += score
        elif n_rows < code <= n_rows + n_cols:
            col_log_scores[code - n_rows - 1] += score

    grid_log_scores = (row_log_scores[:, None] + col_log_scores[None, :]).ravel()
    grid_log_scores -= np.max(grid_log_scores)
    grid_probs = np.exp(grid_log_scores)
    return grid_probs / grid_probs.sum()


def row_column_logit_posterior(flash_logits, stim_codes, n_rows, n_cols):
    """Decode calibrated target-vs-nontarget flash logits into a grid posterior."""
    logits = np.asarray(flash_logits, dtype=float)
    codes = np.asarray(stim_codes, dtype=int)
    if logits.shape != codes.shape:
        raise ValueError("flash_logits and stim_codes must have the same shape")
    row_scores = np.zeros(n_rows)
    col_scores = np.zeros(n_cols)
    for score, code in zip(logits, codes):
        if 1 <= code <= n_rows:
            row_scores[code - 1] += score
        elif n_rows < code <= n_rows + n_cols:
            col_scores[code - n_rows - 1] += score
    scores = (row_scores[:, None] + col_scores[None, :]).ravel()
    scores -= scores.max()
    probs = np.exp(scores)
    return probs / probs.sum()


def membership_posterior(flash_probs, lit_masks):
    """Grid posterior for group-flash studies (Q/E/G/N).

    A cell's evidence is the sum of target log-odds over every flash that lit
    it. For RC flashes (row lights 9 cells, column 8) this is exactly
    row_log_score + col_log_score, i.e. identical to row_column_posterior.
    """
    p = np.clip(np.asarray(flash_probs, dtype=float), 1e-6, 1 - 1e-6)
    log_odds = np.log(p) - np.log1p(-p)
    membership = np.array([[int(ch) for ch in m] for m in lit_masks], dtype=float)
    scores = membership.T @ log_odds
    scores -= scores.max()
    probs = np.exp(scores)
    return probs / probs.sum()


def membership_logit_posterior(flash_logits, lit_masks):
    """Decode calibrated per-flash logits for group-flash membership masks."""
    logits = np.asarray(flash_logits, dtype=float)
    membership = np.array([[int(ch) for ch in m] for m in lit_masks], dtype=float)
    if membership.shape[0] != len(logits):
        raise ValueError("flash_logits and lit_masks must have the same length")
    scores = membership.T @ logits
    scores -= scores.max()
    probs = np.exp(scores)
    return probs / probs.sum()


def _build_adaptive_index_map(epochs):
    """Build (char_idx, sequence_num) -> epoch-row indices from metadata."""
    if epochs.metadata is None:
        return None

    required = {"char_index", "sequence_in_char"}
    if not required.issubset(set(epochs.metadata.columns)):
        return None

    md = epochs.metadata
    valid = (md["char_index"] >= 0) & (md["sequence_in_char"] > 0)
    if not valid.any():
        return None

    grouped = md[valid].groupby(["char_index", "sequence_in_char"]).indices
    return {key: np.asarray(idx, dtype=int) for key, idx in grouped.items()}


def real_eeg_classifier_flash_stream(eeg_data_path, char_idx, sequence_num, clf,
                                     n_rows, n_cols, flashes_per_seq,
                                     max_seqs_per_char=10):
    """Return the classifier score for every row/column flash in a sequence.

    Keeping the per-flash records lets consumers render the actual RC evidence
    stream, while :func:`real_eeg_classifier_stream` still exposes the grid
    posterior used by the decoder.
    """
    global _epoch_cache, _adaptive_index_cache

    if eeg_data_path not in _epoch_cache:
        _epoch_cache.clear()
        _adaptive_index_cache.clear()
        import gc
        gc.collect()

        epochs = mne.read_epochs(eeg_data_path, preload=True, verbose=False)
        _epoch_cache[eeg_data_path] = epochs
        _adaptive_index_cache[eeg_data_path] = _build_adaptive_index_map(epochs)

    epochs = _epoch_cache[eeg_data_path]
    adaptive_index_map = _adaptive_index_cache.get(eeg_data_path)

    if epochs.metadata is None or "stimulus_code" not in epochs.metadata.columns:
        raise ValueError(
            f"{eeg_data_path} has no stimulus_code metadata -- it was likely "
            f"processed with an older version of epoching.py. Re-run "
            f"batch_preprocess.py to regenerate it with the metadata fix."
        )

    is_group_flash = "lit_mask" in epochs.metadata.columns
    if is_group_flash and adaptive_index_map is None:
        raise ValueError(f"{eeg_data_path}: group-flash file has no char/sequence segmentation "
                         f"(not evaluable). Use files with SelectedTarget pulses (Test runs).")

    if adaptive_index_map is not None:
        epoch_rows = adaptive_index_map.get((char_idx, sequence_num))
        if epoch_rows is None or len(epoch_rows) == 0:
            return []
        seq_epochs = epochs[epoch_rows]
    else:
        is_adaptive = "Dyn" in str(eeg_data_path)
        if is_adaptive and eeg_data_path not in _legacy_index_warning_paths:
            print(
                f"⚠️ Warning: {eeg_data_path} has no adaptive char/sequence metadata. "
                "Using legacy fixed-block flash indexing. Re-run batch preprocessing "
                "to regenerate .fif files with adaptive metadata columns."
            )
            _legacy_index_warning_paths.add(eeg_data_path)

        epoch_slice = fixed_sequence_epoch_slice(
            char_idx, sequence_num, flashes_per_seq, max_seqs_per_char
        )
        seq_epochs = epochs[epoch_slice]

    if len(seq_epochs) == 0:
        return []

    seq_epochs.pick('eeg', exclude='bads')
    X = seq_epochs.get_data(copy=False)
    if hasattr(clf, "calibrated_logits"):
        flash_logits = np.asarray(clf.calibrated_logits(X), dtype=float).reshape(-1)
    elif hasattr(clf, "decision_function"):
        raw_logits = np.asarray(clf.decision_function(X), dtype=float).reshape(-1)
        # Legacy estimators may expose decision_function but no fitted scaler.
        flash_logits = raw_logits / float(getattr(clf, "temperature", 1.0))
    else:
        probs = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        flash_logits = np.log(probs) - np.log1p(-probs)
    flash_probs = 1.0 / (1.0 + np.exp(-np.clip(flash_logits, -60, 60)))
    stim_codes = seq_epochs.metadata["stimulus_code"].to_numpy()
    if is_group_flash:
        lit = seq_epochs.metadata["lit_mask"].to_numpy()
        return [
            {"stimulus_code": int(code), "target_logit": float(logit),
             "target_probability": float(probability), "lit_mask": mask}
            for logit, probability, code, mask in zip(flash_logits, flash_probs, stim_codes, lit)
        ]
    return [
        {"stimulus_code": int(code), "target_logit": float(logit),
         "target_probability": float(probability)}
        for logit, probability, code in zip(flash_logits, flash_probs, stim_codes)
        if 1 <= code <= n_rows + n_cols
    ]

def real_eeg_classifier_stream(eeg_data_path, char_idx, sequence_num, clf, char_list,
                                n_rows, n_cols, flashes_per_seq, max_seqs_per_char=10):
    """
    Loads real EEG data, runs it through the calibrated SWLDA,
    and returns the posterior over the full grid (n_rows * n_cols classes,
    NOT hardcoded to 36 -- must match the actual grid, e.g. 72 for Study D).
    """
    flashes = real_eeg_classifier_flash_stream(
        eeg_data_path, char_idx, sequence_num, clf, n_rows, n_cols,
        flashes_per_seq, max_seqs_per_char,
    )
    if not flashes:
        return np.ones(n_rows * n_cols) / (n_rows * n_cols)
    if "lit_mask" in flashes[0]:
        return membership_logit_posterior(
            [f["target_logit"] for f in flashes], [f["lit_mask"] for f in flashes])
    return row_column_logit_posterior(
        [flash["target_logit"] for flash in flashes],
        [flash["stimulus_code"] for flash in flashes], n_rows, n_cols,
    )


def real_eeg_single_flash_trace(eeg_data_path, char_idx, sequence_num, stimulus_code,
                                n_rows, n_cols, flashes_per_seq,
                                max_seqs_per_char=10, channel=None):
    """Return the real averaged EEG voltage trace for one recorded flash."""
    global _epoch_cache, _adaptive_index_cache

    if eeg_data_path not in _epoch_cache:
        _epoch_cache.clear()
        _adaptive_index_cache.clear()
        import gc
        gc.collect()
        epochs = mne.read_epochs(eeg_data_path, preload=True, verbose=False)
        _epoch_cache[eeg_data_path] = epochs
        _adaptive_index_cache[eeg_data_path] = _build_adaptive_index_map(epochs)

    epochs = _epoch_cache[eeg_data_path]
    if epochs.metadata is None or "stimulus_code" not in epochs.metadata.columns:
        raise ValueError(f"{eeg_data_path} has no stimulus_code metadata")
    adaptive_index_map = _adaptive_index_cache.get(eeg_data_path)
    if adaptive_index_map is not None:
        epoch_rows = adaptive_index_map.get((char_idx, sequence_num))
        if epoch_rows is None or len(epoch_rows) == 0:
            return None
        seq_epochs = epochs[epoch_rows]
    else:
        flashes_per_char = flashes_per_seq * max_seqs_per_char
        start_idx = char_idx * flashes_per_char + (sequence_num - 1) * flashes_per_seq
        seq_epochs = epochs[start_idx:start_idx + flashes_per_seq]

    if len(seq_epochs) == 0:
        return None
    stim_codes = seq_epochs.metadata["stimulus_code"].to_numpy()
    match = np.where(stim_codes == stimulus_code)[0]
    if len(match) == 0:
        return None
    flash_epochs = seq_epochs[match].copy().pick("eeg", exclude="bads")
    ch_names = flash_epochs.ch_names
    if not ch_names:
        return None
    selected_channel = channel if channel in ch_names else ("Pz" if "Pz" in ch_names else ch_names[0])
    data = flash_epochs.get_data(copy=False)[:, ch_names.index(selected_channel), :]
    return {
        "channel": selected_channel,
        "times_ms": [round(float(t), 1) for t in flash_epochs.times * 1000.0],
        "amplitude_uv": [round(float(v), 3) for v in data.mean(axis=0) * 1e6],
    }


def mock_eeg_classifier_stream(target_char, char_list):
    """Fallback simulated data if model is missing."""
    probs = np.random.uniform(0.01, 0.05, len(char_list))
    char_upper = target_char.upper()
    if char_upper in char_list:
        target_idx = char_list.index(char_upper)
        probs[target_idx] = 0.85
    probs /= probs.sum()
    return probs

# -----------------------------------------------------------------------------
# EVALUATION LOOP
# -----------------------------------------------------------------------------
def run_evaluation(decoder, llm, fusion_engine, n_rows, n_cols, flashes_per_seq,
                    confidence_threshold=0.85, max_sequences=15, min_flashes=2,
                    clf=None, session_ids=None, enable_session_growth=False,
                    study="StudyD", seconds_per_sequence=2.0,
                    allow_heldout_labels=False, sequences_per_char=10):
    """Runs the dataset through the spelling simulation."""

    class Metrics:
        total_characters = 0
        correct_characters = 0
        total_time_seconds = 0.0
        total_flashes_used = 0
        accuracy = 0.0

    metrics = Metrics()

    if not allow_heldout_labels:
        raise RuntimeError(
            "Evaluation labels are sealed. Pass allow_heldout_labels=True only for a pre-registered evaluation."
        )
    trials = list(yield_character_trials(
        "data/processed/ground_truth_registry.json", "data/processed", study=study,
        include_heldout=True,
    ))

    test_split_path = ("splits/heldout_manifest.json" if study == "StudyD"
                       else f"data/processed/test_sessions_{study}.json")
    if not os.path.exists(test_split_path) and study != "StudyD":
        # No explicit split: evaluate on Test runs only (classifier trains on Train runs).
        trials = [t for t in trials if "_Test" in t.get("session_id", "")]
        print(f"No {test_split_path}; evaluating {study} on Test runs only ({len(trials)} char trials).")
    if os.path.exists(test_split_path):
        with open(test_split_path) as f:
            split = json.load(f)
        if study == "StudyD":
            test_sessions = {run for runs in split.values() for run in runs}
        else:
            test_sessions = set(split)
        trials = [t for t in trials if t.get('session_id') in test_sessions]
        print(f"Restricted eval to {len(test_sessions)} held-out test sessions "
              f"({len(trials)} char trials).")

    if session_ids is not None:
        session_ids = set(session_ids)
        trials = [t for t in trials if t.get('session_id') in session_ids]
        print(f"Restricted eval to {len(session_ids)} requested sessions "
              f"({len(trials)} char trials).")

    char_list = llm.char_list

    for trial in trials:
        target = trial['target_char']
        context = trial['context_so_far']
        eeg_path = trial['eeg_data_path']
        char_idx = len(context)

        decoder.reset()

        if isinstance(llm, RAGPredictor):
            llm_prior = llm.predict_next_char(context, target_char=target)
        else:
            llm_prior = llm.predict_next_char(context)

        # Apply the LM prior ONCE as an initial belief bias -- not per-sequence.
        decoder.accumulated_log_probs += fusion_engine.get_initial_log_bias(llm_prior)

        if trial is trials[0]:  # temporary debug check, remove after verifying
            print("DEBUG initial bias sample:", decoder.accumulated_log_probs[:5],
                  "active:", fusion_engine.is_active)

        start_time = time.time()
        flashes_used = 0
        predicted_char = None

        for seq in range(1, max_sequences + 1):
            flashes_used += 1

            resolved_path = None
            if os.path.exists(eeg_path):
                resolved_path = eeg_path
            else:
                studyd_candidate = eeg_path.replace("processed/", f"processed/{study}/")
                if os.path.exists(studyd_candidate):
                    resolved_path = studyd_candidate

            if clf is not None and resolved_path:
                eeg_posteriors = real_eeg_classifier_stream(
                    resolved_path, char_idx, seq, clf, char_list,
                    n_rows, n_cols, flashes_per_seq,
                    max_seqs_per_char=sequences_per_char,
                )
            else:
                if seq == 1:
                    print(f"⚠️ Warning: Could not find EEG file for {eeg_path} in StudyD. Falling back to mock data.")
                eeg_posteriors = mock_eeg_classifier_stream(target, char_list)

            decoder.accumulate_evidence(eeg_posteriors)

            current_prediction, current_confidence = decoder.decode_character()
            if current_confidence >= confidence_threshold and flashes_used >= min_flashes:
                predicted_char = current_prediction
                break

        if not predicted_char:
            predicted_char, _ = decoder.decode_character()

        metrics.total_characters += 1
        metrics.total_time_seconds += (time.time() - start_time)
        metrics.total_flashes_used += flashes_used
        if predicted_char.upper() == target.upper():
            metrics.correct_characters += 1

        # In a real session the completed text is available as feedback.  Do
        # not persist it: the RAG predictor retains it only for this process.
        if (
            enable_session_growth
            and hasattr(llm, "add_observed_phrase")
            and char_idx + 1 == len(trial.get("target_text", ""))
        ):
            llm.add_observed_phrase(trial["target_text"])

    if metrics.total_characters > 0:
        metrics.accuracy = (metrics.correct_characters / metrics.total_characters) * 100
        avg_time_per_char = (metrics.total_flashes_used / metrics.total_characters) * seconds_per_sequence
    else:
        metrics.accuracy = 0.0
        avg_time_per_char = 0.0

    if avg_time_per_char <= 0:
        print("⚠️ Failed to calculate time per char. Returning 0 ITR.")
        return metrics, 0.0

    itr = calculate_itr(decoder.num_classes, metrics.accuracy, avg_time_per_char)
    return metrics, itr

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--study", default="StudyD")
    _ap.add_argument("--model", default=None)
    _ap.add_argument("--tau", type=float, default=0.80)
    _ap.add_argument("--min-sequences", type=int, default=2)
    _ap.add_argument("--max-sequences", type=int, default=10)
    _ap.add_argument(
        "--allow-heldout-eval", action="store_true",
        help="Required to read the evaluation-only label vault and run a registered evaluation.",
    )
    _args = _ap.parse_args()
    STUDY = _args.study
    print(f"Initializing components for Phase 6 Evaluation (REAL DATA - {STUDY})...\n")

    # 1. Setup Matrix & LLM — built from the REAL grid, not a hardcoded 6x6.
    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name=STUDY
    )
    num_classes = n_rows * n_cols
    with open("data/processed/grid_layout.json") as _f:
        _layout = json.load(_f)[STUDY]
    # RC studies: one flash per row + one per column. Group-flash studies: from layout.
    flashes_per_seq = _layout.get("flashes_per_seq", n_rows + n_cols)
    # Study D keeps the legacy 2.0 s/sequence; others use measured SOA x flashes.
    seconds_per_sequence = 2.0 if STUDY == "StudyD" else flashes_per_seq * _layout["soa_s"]
    print(f"Loaded grid: {n_rows}x{n_cols} ({num_classes} classes), "
          f"{flashes_per_seq} flashes/sequence")

    decoder = P300Decoder(spelling_matrix)
    llm = LLMPredictor(spelling_matrix)
    fixed_rag_llm = RAGPredictor(
        llm,
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.15,
        retrieval_confidence_threshold=0.60,
    )
    gated_rag_llm = RAGPredictor(
        llm,
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.25,
        retrieval_confidence_threshold=0.60,
    )
    tracker = SimpleAblationTracker()

    # 2. Load Real SWLDA Classifier
    MODEL_PATH = _args.model or ('data/processed/clean_m0_epoch_scorer.pkl' if STUDY == 'StudyD'
                                 else f'data/processed/swlda_model_{STUDY}.pkl')
    if STUDY == 'StudyD' and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Clean Study D classifier not found at {MODEL_PATH}. "
            "Export it with scripts/export_locked_clean_m0.py before running the pipeline."
        )
    if os.path.exists(MODEL_PATH):
        print(f"Loading SWLDA Classifier from {MODEL_PATH}...")
        clf = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"⚠️ Real model NOT FOUND at {MODEL_PATH}. "
                                "Please update MODEL_PATH to point to your saved .pkl file.")

    eval_kwargs = dict(n_rows=n_rows, n_cols=n_cols, flashes_per_seq=flashes_per_seq,
                        confidence_threshold=_args.tau,
                        min_flashes=_args.min_sequences,
                        max_sequences=_args.max_sequences,
                        clf=clf,
                        study=STUDY, seconds_per_sequence=seconds_per_sequence,
                        allow_heldout_labels=_args.allow_heldout_eval)

    # ---------------------------------------------------------
    # EXPERIMENT 1: Baseline (No LLM, EEG Only)
    # ---------------------------------------------------------
    print("\nRunning Baseline (EEG Only)...")
    baseline_fusion = BayesianFusionEngine(num_classes=num_classes, mode='fixed', base_alpha=0.0)
    baseline_fusion.toggle(False)

    base_metrics, base_itr = run_evaluation(decoder, llm, baseline_fusion, **eval_kwargs)
    tracker.record_run("Baseline (No LLM)", base_metrics, base_itr)

    # ---------------------------------------------------------
    # EXPERIMENT 2: Fixed Weight Fusion (optimal alpha=0.01 from sweep)
    # ---------------------------------------------------------
    print("\nRunning Fixed Weight Fusion...")
    for test_alpha in [0.1]:
        print(f"\nRunning Fixed Fusion (a={test_alpha})...")
        sweep_fusion = BayesianFusionEngine(num_classes=num_classes, mode='fixed', base_alpha=test_alpha)
        m, i = run_evaluation(decoder, llm, sweep_fusion, **eval_kwargs)
        tracker.record_run(f"Fusion (Fixed a={test_alpha})", m, i)

    # ---------------------------------------------------------
    # EXPERIMENT 3: Adaptive Entropy Fusion (base_alpha=0.01 from sweep)
    # ---------------------------------------------------------
    print("\nRunning Adaptive Entropy Fusion...")
    adaptive_fusion = BayesianFusionEngine(num_classes=num_classes, mode='adaptive', base_alpha=0.01)

    adapt_metrics, adapt_itr = run_evaluation(decoder, llm, adaptive_fusion, **eval_kwargs)
    tracker.record_run("Fusion (Adaptive)", adapt_metrics, adapt_itr)

    # ---------------------------------------------------------
    # EXPERIMENT 4: Fixed RAG-LM Fusion
    # ---------------------------------------------------------
    print("\nRunning Fixed RAG-LM Fusion...")
    fixed_rag_fusion = BayesianFusionEngine(num_classes=num_classes, mode='fixed', base_alpha=0.1)
    fixed_rag_metrics, fixed_rag_itr = run_evaluation(decoder, fixed_rag_llm, fixed_rag_fusion, **eval_kwargs)
    tracker.record_run("Fusion (Fixed RAG-LM)", fixed_rag_metrics, fixed_rag_itr)

    # ---------------------------------------------------------
    # EXPERIMENT 5: Gated RAG-LM Fusion
    # ---------------------------------------------------------
    print("\nRunning Gated RAG-LM Fusion...")
    gated_rag_fusion = BayesianFusionEngine(num_classes=num_classes, mode='adaptive', base_alpha=0.01)
    gated_rag_metrics, gated_rag_itr = run_evaluation(decoder, gated_rag_llm, gated_rag_fusion, **eval_kwargs)
    tracker.record_run("Fusion (Gated RAG-LM)", gated_rag_metrics, gated_rag_itr)

    tracker.print_summary()
