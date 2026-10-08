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


def test_update_adds_only_nonempty_decoded_text_to_session_bank(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_text("text\nalpha beta\n", encoding="utf-8")
    predictor = RAGPredictor(UniformPredictor(), phrase_bank_path=path, subject_id="Q_01")
    initial = len(predictor.subject_phrases)
    assert predictor.update("") is False
    assert predictor.update("  ") is False
    assert predictor.update("alpha beta") is True
    assert len(predictor.subject_phrases) == initial + 1
    assert predictor.subject_phrases[-1].source == "observed_subject_session"
    assert predictor.update("gamma") is True
    assert predictor.subject_token_count == sum(len(p.text.split()) for p in predictor.subject_phrases)


def test_update_without_personalized_subject_bank_is_noop(tmp_path):
    predictor = RAGPredictor(UniformPredictor(), phrase_bank_path=tmp_path / "empty.csv")
    assert predictor.update("decoded words") is False
    assert predictor.subject_phrases == []


def test_token_mode_recognizes_multiword_and_special_key_aliases(tmp_path):
    path = tmp_path / "bank.csv"
    path.write_text("text\na b page down\na b backspace\n", encoding="utf-8")

    class SpecialPredictor(UniformPredictor):
        special_key_map = {"PgDn": "page down", "Bs": "backspace"}
        char_list = ["A", "Sp", "Sleep", "PgUp", "PgDn", "Bs"]

    predictor = RAGPredictor(SpecialPredictor(), phrase_bank_path=path, token_mode=True)
    assert predictor._char_to_grid_index("page down") == 4
    assert predictor._char_to_grid_index("backspace") == 5
    assert predictor._matching_phrases("a b")[0].next_char in {"page down", "backspace"}


def test_empty_oov_and_adversarial_banks_return_finite_normalized_priors(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("text\n", encoding="utf-8")
    predictor = RAGPredictor(UniformPredictor(), phrase_bank_path=empty, rag_weight=1.0)
    prior = predictor.predict_next_char("unseen context")
    assert np.isfinite(prior).all()
    assert np.isclose(prior.sum(), 1.0)

    adversarial = tmp_path / "adversarial.csv"
    adversarial.write_text("text,weight\naaa,-1000000\naaa,1e300\n", encoding="utf-8")
    robust = RAGPredictor(UniformPredictor(), phrase_bank_path=adversarial, rag_weight=1.0)
    prior = robust.predict_next_char("aa")
    assert np.isfinite(prior).all()
    assert np.all(prior >= 0)
    assert np.isclose(prior.sum(), 1.0)
