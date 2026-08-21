import sys
sys.path.insert(0, "src")
from src.preprocessing.build_session_sequence import yield_character_trials
from src.models.llm_predictor import LLMPredictor
import numpy as np, json

def load_spelling_matrix(grid_layout_path, study_name="StudyD"):
    with open(grid_layout_path) as f:
        layouts = json.load(f)
    info = layouts[study_name]
    n_rows, n_cols = info["n_rows"], info["n_cols"]
    matrix = np.full((n_rows, n_cols), "", dtype=object)
    for label, (row, col) in info["grid_map"].items():
        matrix[row - 1, col - 1] = label
    return matrix, n_rows, n_cols

spelling_matrix, n_rows, n_cols = load_spelling_matrix("data/processed/grid_layout.json", "StudyD")
llm = LLMPredictor(spelling_matrix)
trials = list(yield_character_trials("data/processed/ground_truth_registry.json", "data/processed"))

correct = 0
for t in trials:
    probs = llm.predict_next_char(t['context_so_far'])
    top1 = llm.char_list[np.argmax(probs)]
    if top1.upper() == t['target_char'].upper():
        correct += 1
print(f"LLM top-1 accuracy: {correct/len(trials)*100:.2f}% over {len(trials)} trials")