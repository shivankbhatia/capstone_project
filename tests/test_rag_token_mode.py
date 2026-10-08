import csv

import numpy as np

from src.models.rag_predictor import RAGPredictor


class UniformPredictor:
    char_list = ["A", "Sp", "Sleep", "PgUp"]

    def predict_next_char(self, context):
        return np.full(len(self.char_list), 1 / len(self.char_list))


def test_token_mode_retrieves_complete_multichar_grid_keys(tmp_path):
    path = tmp_path / "bank.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "source", "weight", "category"))
        writer.writeheader()
        writer.writerow({"text": "a b sleep sp pgup", "source": "test", "weight": 1, "category": "test"})
    predictor = RAGPredictor(
        UniformPredictor(), phrase_bank_path=path, token_mode=True,
        retrieval_confidence_threshold=0.0,
    )
    matches = predictor._matching_phrases("a b")
    assert matches
    assert matches[0].next_char == "sleep"
    assert predictor._char_to_grid_index(matches[0].next_char) == 2
    space_matches = predictor._matching_phrases("a b sleep")
    assert space_matches[0].next_char == "sp"
    assert predictor._char_to_grid_index(space_matches[0].next_char) == 1
