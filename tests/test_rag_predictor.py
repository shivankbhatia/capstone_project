import csv

import numpy as np

from src.models.rag_predictor import RAGPredictor


class _BasePredictor:
    char_list = ["a", "b", "c", "Sp"]

    def predict_next_char(self, context_so_far):
        return np.full(4, 0.25)


def _write_bank(path, texts):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text"])
        writer.writeheader()
        writer.writerows({"text": text} for text in texts)


def test_subject_blend_is_continuous_and_uses_cold_start_ramp(tmp_path):
    global_bank = tmp_path / "phrase_bank_global.csv"
    subject_dir = tmp_path / "by_subject"
    subject_dir.mkdir()
    _write_bank(global_bank, ["able"])
    _write_bank(subject_dir / "phrase_bank_S1.csv", ["able", "able"])
    predictor = RAGPredictor(
        _BasePredictor(), phrase_bank_path=global_bank, subject_id="S1",
        subject_only=True, min_subject_phrases=1, rag_weight=0.5,
        subject_confidence_threshold=0.2, subject_gate_sharpness=10,
        sufficiency_midpoint_tokens=2, sufficiency_sharpness=1,
    )

    predictor.predict_next_char("a")
    diagnostics = predictor.last_diagnostics

    assert 0.0 < diagnostics.subject_confidence_weight < 1.0
    assert 0.0 < diagnostics.subject_coldstart_weight < 1.0
    assert 0.0 < diagnostics.subject_blend_weight < 1.0
    assert diagnostics.subject_blend_weight == (
        predictor.rag_weight * predictor.personalization_bonus
        * predictor.subject_weight * diagnostics.subject_confidence_weight
        * diagnostics.subject_coldstart_weight
    )


def test_subject_bigram_backoff_is_used_when_no_word_candidate_exists(tmp_path):
    global_bank = tmp_path / "phrase_bank_global.csv"
    subject_dir = tmp_path / "by_subject"
    subject_dir.mkdir()
    _write_bank(global_bank, ["able"])
    _write_bank(subject_dir / "phrase_bank_S1.csv", ["azb", "azc"])
    predictor = RAGPredictor(
        _BasePredictor(), phrase_bank_path=global_bank, subject_id="S1",
        subject_only=True, min_subject_phrases=1, rag_weight=0.5,
        sufficiency_midpoint_tokens=2, sufficiency_sharpness=1,
        bigram_min_count=1,
        bigram_max_normalized_entropy=1.0,
    )

    prior = predictor.predict_next_char("z")

    assert predictor.last_diagnostics.subject_source == "char_backoff"
    assert predictor.char_backoff_count == 1
    assert predictor.blend_history[-1]["subject_source"] == "char_backoff"
    assert np.isclose(prior.sum(), 1.0)


def test_bigram_count_gate_falls_back_to_base_prior(tmp_path):
    global_bank = tmp_path / "phrase_bank_global.csv"
    subject_dir = tmp_path / "by_subject"
    subject_dir.mkdir()
    _write_bank(global_bank, ["able"])
    _write_bank(subject_dir / "phrase_bank_S1.csv", ["azb", "azc"])
    predictor = RAGPredictor(
        _BasePredictor(), phrase_bank_path=global_bank, subject_id="S1",
        subject_only=True, min_subject_phrases=1, rag_weight=0.5,
        bigram_min_count=2,
    )

    prior = predictor.predict_next_char("z")

    assert predictor.last_diagnostics.subject_source == "none"
    assert predictor.source_counts["below_min_count"] == 1
    np.testing.assert_allclose(prior, np.full(4, 0.25))
