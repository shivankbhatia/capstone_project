import numpy as np

from src.models.rag_predictor import RAGPredictor


class UniformPredictor:
    def __init__(self):
        self.char_list = ["A", "B", "C", "D", "E", "H", "I", "L", "N", "O", "P", "S", "T", "Y", "Sp"]

    def predict_next_char(self, context_so_far):
        return np.ones(len(self.char_list)) / len(self.char_list)


def test_rag_predictor_boosts_prefix_compatible_next_char():
    predictor = RAGPredictor(
        UniformPredictor(),
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.5,
        retrieval_confidence_threshold=0.1,
    )

    prior = predictor.predict_next_char("HE")
    h_idx = predictor.char_list.index("H")
    l_idx = predictor.char_list.index("L")

    assert predictor.last_diagnostics.rag_enabled
    assert predictor.last_diagnostics.partial_word == "he"
    assert prior[l_idx] > prior[h_idx]
    assert np.isclose(prior.sum(), 1.0)


def test_rag_predictor_falls_back_to_base_prior_without_matches():
    predictor = RAGPredictor(
        UniformPredictor(),
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.5,
        retrieval_confidence_threshold=0.1,
    )

    prior = predictor.predict_next_char("ZX")

    assert not predictor.last_diagnostics.rag_enabled
    assert predictor.last_diagnostics.reason == "no_matches"
    assert np.allclose(prior, np.ones(len(predictor.char_list)) / len(predictor.char_list))
