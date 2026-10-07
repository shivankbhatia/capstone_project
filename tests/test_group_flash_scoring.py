"""Tests for group-flash support using the recorded Study Q sample."""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import run_pipeline as rp
except Exception:  # heavy LLM dependencies are irrelevant to scoring helpers
    for name, attr in [("src.models.llm_predictor", "LLMPredictor"),
                       ("src.models.rag_predictor", "RAGPredictor"),
                       ("src.models.fusion", "BayesianFusionEngine")]:
        module = types.ModuleType(name)
        setattr(module, attr, object)
        sys.modules[name] = module
    import run_pipeline as rp

from src.data.batch_preprocess import parse_bigp3bci_edf

N_ROWS, N_COLS = 9, 8
N_CELLS = N_ROWS * N_COLS
STUDY_Q_SAMPLE = ROOT / "samples" / "StudyQ.edf"
STUDY_D_SAMPLE = (ROOT / "data" / "raw" / "bigP3BCI_dataset" / "bigP3BCI-data"
                  / "StudyD" / "D_01" / "SE001" / "Test" / "Dyn"
                  / "D_01_SE001_Dyn_Test01.edf")


def test_membership_matches_row_column_exactly():
    rng = np.random.default_rng(1)
    codes, masks, probs = [], [], rng.uniform(0.02, 0.98, 17 * 3)
    for _ in range(3):
        for code in range(1, 18):
            lit = np.zeros(N_CELLS, int)
            if code <= N_ROWS:
                lit[(code - 1) * N_COLS:code * N_COLS] = 1
            else:
                lit[(code - N_ROWS - 1)::N_COLS] = 1
            codes.append(code)
            masks.append("".join(map(str, lit)))
    row_column = rp.row_column_posterior(probs, codes, N_ROWS, N_COLS)
    membership = rp.membership_posterior(probs, masks)
    assert np.allclose(row_column, membership, atol=1e-12)
    logits = np.log(probs) - np.log1p(-probs)
    assert np.allclose(
        rp.row_column_logit_posterior(logits, codes, N_ROWS, N_COLS),
        rp.membership_logit_posterior(logits, masks), atol=1e-12,
    )


def test_recorded_study_q_file_end_to_end():
    epochs, text, info = parse_bigp3bci_edf(str(STUDY_Q_SAMPLE))

    assert text == "DRIVING"
    assert info["group_flash"]
    assert info["flashes_per_seq"] == 12
    assert info["soa_s"] == 0.125
    assert info["agreement"]["is_adaptive"]
    assert info["agreement"]["n_characters_detected"] == 7

    metadata = epochs.metadata
    assert len(metadata) == 504
    assert set(metadata.columns) >= {"lit_mask", "char_index", "sequence_in_char"}
    assert metadata["lit_mask"].str.len().eq(N_CELLS).all()
    assert metadata["lit_mask"].str.count("1").eq(6).all()
    assert metadata.groupby("char_index").size().eq(72).all()
    assert metadata.groupby("char_index")["sequence_in_char"].max().eq(6).all()
    assert metadata.groupby(["char_index", "sequence_in_char"]).size().eq(12).all()


def test_recorded_study_d_file_uses_row_column_path():
    epochs, text, info = parse_bigp3bci_edf(str(STUDY_D_SAMPLE))

    assert text == "VISUAL"
    assert not info["group_flash"]
    assert info["flashes_per_seq"] == N_ROWS + N_COLS
    assert info["soa_s"] is None
    assert info["agreement"]["is_adaptive"]
    assert info["agreement"]["n_characters_detected"] == 6

    metadata = epochs.metadata
    assert len(metadata) == 243
    assert "lit_mask" not in metadata.columns
    assert set(metadata.columns) == {"stimulus_code", "stimulus_type", "char_index", "sequence_in_char"}
    assert metadata["stimulus_code"].between(1, N_ROWS + N_COLS).all()
