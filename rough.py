import json
from src.preprocessing.build_session_sequence import yield_character_trials
from src.models.llm_predictor import LLMPredictor
import numpy as np

def load_spelling_matrix(grid_layout_path, study_name="StudyD"):
    with open(grid_layout_path) as f:
        layouts = json.load(f)
    if study_name not in layouts:
        raise ValueError(f"No grid layout recorded for {study_name} in {grid_layout_path}.")
    info = layouts[study_name]
    n_rows, n_cols = info["n_rows"], info["n_cols"]
    grid_map = info["grid_map"]
    matrix = np.full((n_rows, n_cols), "", dtype=object)
    for label, (row, col) in grid_map.items():
        matrix[row - 1, col - 1] = label
    if np.any(matrix == ""):
        missing = np.argwhere(matrix == "")
        print(f"WARNING: {len(missing)} grid cell(s) have no label.")
    return matrix, n_rows, n_cols


spelling_matrix, n_rows, n_cols = load_spelling_matrix("data/processed/grid_layout.json", "StudyD")
llm = LLMPredictor(spelling_matrix)

trials = list(yield_character_trials("data/processed/ground_truth_registry.json", "data/processed"))

# --- Top-1 accuracy over all trials ---
correct = 0
for t in trials[:50]:
    probs = llm.predict_next_char(t['context_so_far'])
    top1 = llm.char_list[np.argmax(probs)]
    print(f"{t['context_so_far']!r} -> pred {top1!r}, target {t['target_char']!r}, match={top1.upper()==t['target_char'].upper()}")
print(f"LLM top-1 accuracy: {correct/len(trials)*100:.2f}% over {len(trials)} trials")

# --- Sample trial breakdown (same trial as before, for direct comparison) ---
t = trials[10]
print(f"\ncontext: {t['context_so_far']!r}  target: {t['target_char']!r}")
probs = llm.predict_next_char(t['context_so_far'])
top5_idx = np.argsort(probs)[::-1][:5]
for i in top5_idx:
    print(f"  {llm.char_list[i]!r}: {probs[i]:.4f}")
