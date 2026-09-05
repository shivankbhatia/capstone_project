import numpy as np

from run_pipeline import row_column_posterior


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
