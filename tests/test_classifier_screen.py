import numpy as np

from scripts.screen_shrinkage_lda import _character_metrics


def test_sequence_decoder_recovers_known_row_column_character():
    codes = np.tile(np.arange(1, 18), 20)
    target = np.isin(codes, [3, 11]).astype(np.uint8)
    logits = np.where(target == 1, 4.0, -4.0)
    metrics = _character_metrics(
        logits,
        target,
        np.full(len(codes), "D_01_SE001_RC_Train01"),
        np.zeros(len(codes), dtype=int),
        np.repeat(np.arange(20), 17),
        codes,
    )
    assert metrics["characters"] == 1
    assert metrics["full_repetition_accuracy"] == 1.0
    assert metrics["locked_stopping_accuracy"] == 1.0
    assert metrics["mean_sequences_at_stop"] == 2.0
