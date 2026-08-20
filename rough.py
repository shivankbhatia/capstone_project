import json
from src.preprocessing.build_session_sequence import yield_character_trials
from src.models.llm_predictor import LLMPredictor
import numpy as np

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


# reuse spelling_matrix/llm already loaded in your run_pipeline.py session, or:
spelling_matrix, n_rows, n_cols = load_spelling_matrix("data/processed/grid_layout.json", "StudyD")
llm = LLMPredictor(spelling_matrix)

trials = list(yield_character_trials("data/processed/ground_truth_registry.json", "data/processed"))

# --- Top-1 accuracy over all trials ---
correct = 0
for t in trials:
    probs = llm.predict_next_char(t['context_so_far'])
    top1 = llm.char_list[np.argmax(probs)]
    if top1.upper() == t['target_char'].upper():
        correct += 1
print(f"LLM top-1 accuracy: {correct/len(trials)*100:.2f}% over {len(trials)} trials")

# --- Sample trial breakdown ---
t = trials[10]
print(f"\ncontext: {t['context_so_far']!r}  target: {t['target_char']!r}")
probs = llm.predict_next_char(t['context_so_far'])
top5_idx = np.argsort(probs)[::-1][:5]
for i in top5_idx:
    print(f"  {llm.char_list[i]!r}: {probs[i]:.4f}")