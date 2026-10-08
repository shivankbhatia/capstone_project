import numpy as np
import pytest
from scipy.special import softmax

from src.evaluation.sequence_scoring import (
    SequentialDecoder,
    aggregate_flash_logits,
    aggregate_membership_logits,
    variable_time_itr,
)
from src.models.decoder import calculate_itr
from scripts.evaluate_classifier_manifest import _sequence_rows


def test_flash_logits_match_row_column_outer_product():
    codes = np.arange(1, 18)
    logits = np.linspace(-0.8, 0.9, 17)
    result = aggregate_flash_logits(logits, codes)
    rows, cols = np.zeros(9), np.zeros(8)
    for score, code in zip(logits, codes):
        if code <= 9:
            rows[code - 1] += score
        else:
            cols[code - 10] += score
    expected = np.outer(softmax(rows), softmax(cols)).ravel()
    np.testing.assert_allclose(result.grid, expected, rtol=1e-12, atol=1e-12)
    assert result.valid_flash_count == 17


def test_flash_logits_require_both_axes_and_ignore_invalid_codes():
    assert aggregate_flash_logits([0.2, 0.4], [1, 2]) is None
    result = aggregate_flash_logits([0.2, np.nan, 0.4], [1, 99, 10])
    assert result.valid_flash_count == 2
    assert result.grid.shape == (72,)


def test_group_membership_logits_match_row_column_grid_posterior():
    n_rows, n_cols = 3, 2
    codes = np.array([1, 2, 3, 4, 5])
    logits = np.array([0.2, -0.4, 0.7, 0.1, -0.2])
    masks = []
    for code in codes:
        mask = np.zeros(n_rows * n_cols, dtype=int)
        if code <= n_rows:
            mask[(code - 1) * n_cols:code * n_cols] = 1
        else:
            mask[code - n_rows - 1::n_cols] = 1
        masks.append(mask)
    np.testing.assert_allclose(
        aggregate_membership_logits(logits, masks, n_rows, n_cols),
        aggregate_flash_logits(logits, codes, n_rows, n_cols).grid,
    )


def test_group_membership_accepts_parser_binary_strings():
    actual = aggregate_membership_logits([1.0, -1.0], ["10", "01"], 1, 2)
    assert actual[0] > actual[1]


def test_shared_stopping_waits_for_minimum_and_uses_threshold():
    decoder = SequentialDecoder(3, tau=0.8, min_sequences=2, max_sequences=4)
    first, stopped = decoder.add([0.9, 0.05, 0.05])
    assert not stopped and first.argmax() == 0
    second, stopped = decoder.add([0.9, 0.05, 0.05])
    assert stopped and second.argmax() == 0
    assert decoder.result()["sequences_used"] == 2


def test_shared_stopping_respects_maximum_without_threshold_crossing():
    decoder = SequentialDecoder(3, tau=0.99, min_sequences=2, max_sequences=2)
    decoder.add([0.34, 0.33, 0.33])
    decoder.add([0.34, 0.33, 0.33])
    assert not decoder.result()["stopped_by_threshold"]
    assert decoder.result()["sequences_used"] == 2
    with pytest.raises(ValueError):
        SequentialDecoder(3, tau=0.8, min_sequences=3, max_sequences=2)


def test_shared_itr_preserves_legacy_formula():
    for accuracy in (0.0, 0.25, 0.8, 1.0):
        assert calculate_itr(72, accuracy * 100, 5.4) == pytest.approx(
            variable_time_itr(72, accuracy, 5.4)
        )


def test_oof_sequence_grouping_keeps_runs_separate_and_includes_index_zero():
    codes = np.tile(np.arange(1, 18), 2)
    run_ids = np.asarray(["run_a"] * 17 + ["run_b"] * 17)
    chars = np.zeros(34, dtype=int)
    seqs = np.zeros(34, dtype=int)
    labels = np.zeros(34, dtype=int)
    labels[0], labels[9] = 1, 1  # run A: A = row 1, column 1.
    labels[17], labels[27] = 1, 1  # run B: B = row 1, column 2.
    logits = np.zeros(34)
    logits[[0, 9, 17, 27]] = 2.0
    by_char, _ = _sequence_rows(
        logits, labels, codes, chars, seqs, list(range(72)), run_ids=run_ids
    )
    assert set(by_char) == {("run_a", 0), ("run_b", 0)}
    assert len(by_char[("run_a", 0)]) == len(by_char[("run_b", 0)]) == 1
    assert not np.allclose(by_char[("run_a", 0)][0][2], by_char[("run_b", 0)][0][2])
