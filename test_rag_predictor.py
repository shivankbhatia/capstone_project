import numpy as np

from src.models.rag_predictor import RAGPredictor


class UniformPredictor:
    def __init__(self):
        self.char_list = [
            "A", "B", "C", "D", "E",
            "H", "I", "L", "N", "O",
            "P", "S", "T", "Y", "Sp"
        ]

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


def test_rag_predictor_uses_phrase_level_context_after_trailing_space():
    predictor = RAGPredictor(
        UniformPredictor(),
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.5,
        retrieval_confidence_threshold=0.6,
    )

    prior = predictor.predict_next_char("I NEED ")
    h_idx = predictor.char_list.index("H")
    l_idx = predictor.char_list.index("L")

    assert predictor.last_diagnostics.rag_enabled
    assert predictor.last_diagnostics.normalized_context == "i need "
    assert predictor.last_diagnostics.matched_context == "i need "
    assert predictor.last_diagnostics.matched_phrases == ["i need help"]
    assert prior[h_idx] > prior[l_idx]


def test_rag_predictor_prefers_longest_phrase_context():
    predictor = RAGPredictor(
        UniformPredictor(),
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.5,
        retrieval_confidence_threshold=0.6,
    )

    prior = predictor.predict_next_char("TURN ON THE L")
    i_idx = predictor.char_list.index("I")
    t_idx = predictor.char_list.index("T")

    assert predictor.last_diagnostics.rag_enabled
    assert predictor.last_diagnostics.matched_context == "turn on the l"
    assert predictor.last_diagnostics.matched_phrases == ["turn on the lights"]
    assert prior[i_idx] > prior[t_idx]


def test_rag_predictor_ranks_ambiguous_prefix_by_context_relevance():
    predictor = RAGPredictor(
        UniformPredictor(),
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.5,
        retrieval_confidence_threshold=0.6,
    )

    prior = predictor.predict_next_char("HEL")
    l_idx = predictor.char_list.index("L")
    p_idx = predictor.char_list.index("P")

    assert predictor.last_diagnostics.rag_enabled
    assert predictor.last_diagnostics.retrieval_confidence >= 0.6
    assert predictor.last_diagnostics.matched_phrases[:2] == [
        "hello world",
        "i need help",
    ]
    assert prior[l_idx] > prior[p_idx]


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
    assert np.allclose(
        prior,
        np.ones(len(predictor.char_list)) / len(predictor.char_list),
    )