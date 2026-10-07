import numpy as np

from run_pipeline import (
    fixed_sequence_epoch_slice, row_column_logit_posterior, row_column_posterior,
)


def test_standard_complete_sequence_selects_target_cell():
    # Two rows (codes 1-2) and two columns (codes 3-4); row 2 / column 1 is
    # the only target pair.
    posterior = row_column_posterior(
        flash_probs=[0.05, 0.90, 0.95, 0.05],
        stim_codes=[1, 2, 3, 4],
        n_rows=2,
        n_cols=2,
    )

    assert posterior.shape == (4,)
    assert posterior.argmax() == 2  # row 2, column 1 in row-major order
    assert np.isclose(posterior.sum(), 1.0)


def test_dynamic_repeated_flash_is_accumulated_not_overwritten():
    # In a dynamic sequence, code 2 is flashed twice. The second observation
    # is weak, but the two flashes together still strongly support row 2.
    posterior = row_column_posterior(
        flash_probs=[0.10, 0.95, 0.60, 0.95, 0.10],
        stim_codes=[1, 2, 2, 3, 4],
        n_rows=2,
        n_cols=2,
    )

    assert posterior.argmax() == 2  # row 2, column 1


def test_logits_path_matches_probability_path():
    probs = np.array([0.05, 0.90, 0.95, 0.05])
    logits = np.log(probs) - np.log1p(-probs)
    from_logits = row_column_logit_posterior(logits, [1, 2, 3, 4], 2, 2)
    from_probs = row_column_posterior(probs, [1, 2, 3, 4], 2, 2)
    assert np.allclose(from_logits, from_probs)


def test_fixed_run_epoch_index_uses_acquisition_sequences_not_stop_limit():
    # Clean Study D RC/Train runs store 10 repetitions per character. The
    # decoder's stop limit is a separate setting and must not shift indexing.
    assert fixed_sequence_epoch_slice(1, 1, 17) == slice(170, 187)
    assert fixed_sequence_epoch_slice(1, 10, 17) == slice(323, 340)
    assert fixed_sequence_epoch_slice(1, 2, 17, sequences_per_char=20) == slice(357, 374)
