import numpy as np

from scripts.screen_shrinkage_lda import _character_metrics
from scripts.build_classifier_cache import _sequence_and_character_indices


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


def test_fixed_rc_cache_uses_ten_sequences_per_character():
    class EpochStub:
        metadata = None

        def __len__(self):
            return 17 * 10 * 6

    seq_idx, char_idx = _sequence_and_character_indices(EpochStub(), "D_01_SE001_RC_Train01")
    assert len(set(char_idx)) == 6
    assert len(set(seq_idx)) == 10
    assert char_idx[17 * 10] == 1
    assert seq_idx[17 * 10] == 0
