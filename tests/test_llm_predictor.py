import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from src.models.llm_predictor import LLMPredictor


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


class _UniformModel:
    def __call__(self, input_ids, attention_mask=None):
        vocab = 256
        return SimpleNamespace(logits=torch.zeros((*input_ids.shape, vocab)))


def _predictor():
    predictor = LLMPredictor.__new__(LLMPredictor)
    predictor.char_list = ["A", "PgUp", "LfAw", "email"]
    predictor.special_key_map = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/llm/special_key_map.json")
        .read_text(encoding="utf-8")
    )
    predictor.tokenizer = _CharTokenizer()
    predictor.model = _UniformModel()
    predictor.bos_token_id = 1
    predictor.grid_matrix = np.asarray(predictor.char_list).reshape(1, -1)
    return predictor


def test_study_d_multi_character_keys_have_explicit_text_mappings():
    predictor = _predictor()
    assert predictor._candidate_text("PgUp") == "page up"
    assert predictor._candidate_text("LfAw") == "left arrow"
    assert predictor._candidate_text("email") == "email"


def test_full_string_scoring_accumulates_all_tokens_and_returns_distribution():
    predictor = _predictor()
    short = predictor._continuation_log_probability("", "a")
    long = predictor._continuation_log_probability("", "page up")
    assert long < short

    probabilities = predictor.predict_next_char("hel")
    assert probabilities.shape == (4,)
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0)
