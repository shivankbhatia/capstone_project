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

def real_eeg_classifier_stream(eeg_data_path, char_idx, sequence_num, clf, char_list,
                                n_rows, n_cols, flashes_per_seq, max_seqs_per_char=15):
    """
    Loads real EEG data, runs it through the calibrated SWLDA,
    and returns the posterior over the full grid (n_rows * n_cols classes,
    NOT hardcoded to 36 -- must match the actual grid, e.g. 72 for Study D).
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

    if adaptive_index_map is not None:
        key = (char_idx, sequence_num)
        epoch_rows = adaptive_index_map.get(key)
        if epoch_rows is None or len(epoch_rows) == 0:
            return np.ones(n_rows * n_cols) / (n_rows * n_cols)
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

        flashes_per_char = flashes_per_seq * max_seqs_per_char
        start_idx = (char_idx * flashes_per_char) + ((sequence_num - 1) * flashes_per_seq)
        end_idx = start_idx + flashes_per_seq
        seq_epochs = epochs[start_idx:end_idx]

    if len(seq_epochs) == 0:
        return np.ones(n_rows * n_cols) / (n_rows * n_cols)

    seq_epochs.pick('eeg', exclude='bads')

    X = seq_epochs.get_data(copy=False)
    flash_probs = clf.predict_proba(X)[:, 1]

    # Row/column identity comes from the preserved stimulus_code metadata,
    # NOT from epochs.events[:, 2] (which only holds the binary
    # target/non-target label used to build the epochs' event_id).
    stim_codes = seq_epochs.metadata["stimulus_code"].to_numpy()

    row_probs = np.ones(n_rows)
    col_probs = np.ones(n_cols)

    for prob, code in zip(flash_probs, stim_codes):
        if 1 <= code <= n_rows:
            row_probs[code - 1] = prob
        elif n_rows < code <= n_rows + n_cols:
            col_probs[code - n_rows - 1] = prob

    grid_probs = np.outer(row_probs, col_probs).flatten()
    sum_probs = np.sum(grid_probs)
    if sum_probs > 0:
        normalized_probs = grid_probs / sum_probs
    else:
        normalized_probs = np.ones(n_rows * n_cols) / (n_rows * n_cols)
    return normalized_probs


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
                    confidence_threshold=0.85, max_sequences=15, min_flashes=2, clf=None):
    """Runs the dataset through the spelling simulation."""

    class Metrics:
        total_characters = 0
        correct_characters = 0
        total_time_seconds = 0.0
        total_flashes_used = 0
        accuracy = 0.0

    metrics = Metrics()

    trials = list(yield_character_trials("data/processed/ground_truth_registry.json", "data/processed"))
    char_list = llm.char_list

    for trial in trials:
        target = trial['target_char']
        context = trial['context_so_far']
        eeg_path = trial['eeg_data_path']
        char_idx = len(context)

        decoder.reset()
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
                studyd_candidate = eeg_path.replace("processed/", "processed/StudyD/")
                if os.path.exists(studyd_candidate):
                    resolved_path = studyd_candidate

            if clf is not None and resolved_path:
                eeg_posteriors = real_eeg_classifier_stream(
                    resolved_path, char_idx, seq, clf, char_list,
                    n_rows, n_cols, flashes_per_seq, max_seqs_per_char=max_sequences
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

    if metrics.total_characters > 0:
        metrics.accuracy = (metrics.correct_characters / metrics.total_characters) * 100
        avg_time_per_char = (metrics.total_flashes_used / metrics.total_characters) * 2.0
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
    print("Initializing components for Phase 6 Evaluation (REAL DATA - StudyD only)...\n")

    # 1. Setup Matrix & LLM — built from the REAL grid, not a hardcoded 6x6.
    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name="StudyD"
    )
    num_classes = n_rows * n_cols
    # Study D's RC (row-column) condition: one flash per row + one per column.
    flashes_per_seq = n_rows + n_cols
    print(f"Loaded grid: {n_rows}x{n_cols} ({num_classes} classes), "
          f"{flashes_per_seq} flashes/sequence")

    decoder = P300Decoder(spelling_matrix)
    llm = LLMPredictor(spelling_matrix)
    fixed_rag_llm = RAGPredictor(
        llm,
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.25,
        retrieval_confidence_threshold=0.0,
    )
    gated_rag_llm = RAGPredictor(
        llm,
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.25,
        retrieval_confidence_threshold=0.60,
    )
    tracker = SimpleAblationTracker()

    # 2. Load Real SWLDA Classifier
    MODEL_PATH = 'data/processed/swlda_model.pkl'
    if os.path.exists(MODEL_PATH):
        print(f"Loading SWLDA Classifier from {MODEL_PATH}...")
        clf = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"⚠️ Real model NOT FOUND at {MODEL_PATH}. "
                                "Please update MODEL_PATH to point to your saved .pkl file.")

    eval_kwargs = dict(n_rows=n_rows, n_cols=n_cols, flashes_per_seq=flashes_per_seq,
                        confidence_threshold=0.85, clf=clf)

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