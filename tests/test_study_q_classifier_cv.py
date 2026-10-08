import numpy as np

from scripts.nested_study_q_calibration_fusion import _eval_policy, _fit_temperature
from scripts.run_study_q_classifier_cv import _character_metrics


def test_group_mask_character_scoring_recovers_true_cell_across_sequences():
    rng = np.random.default_rng(19)
    target = 37
    logits, labels, masks, seqs, chars, runs, subjects = [], [], [], [], [], [], []
    for sequence in range(4):
        order = rng.permutation(72)
        for start in range(0, 72, 6):
            mask = np.zeros(72, dtype=np.uint8)
            mask[order[start:start + 6]] = 1
            positive = int(mask[target])
            masks.append(mask)
            labels.append(positive)
            logits.append(3.0 if positive else -1.0)
            seqs.append(sequence)
            chars.append(0)
            runs.append("Q_07_SE001_Grey-to-White_Train01")
            subjects.append("Q_07")
    result = _character_metrics(
        np.asarray(logits),
        {
            "y": np.asarray(labels), "lit_mask": np.asarray(masks),
            "seq_idx": np.asarray(seqs), "char_idx": np.asarray(chars),
            "run_id": np.asarray(runs), "subject": np.asarray(subjects),
        },
    )
    assert result["characters"] == 1
    assert result["invalid_target_intersections"] == 0
    assert result["full_available_sequence_accuracy"] == 1.0


def test_nested_sequence_temperature_and_locked_stopping_use_group_masks():
    rng = np.random.default_rng(21)
    target = 37
    logits, labels, masks, seqs, chars, runs, subjects, sessions = [], [], [], [], [], [], [], []
    for sequence in range(4):
        order = rng.permutation(72)
        for start in range(0, 72, 6):
            mask = np.zeros(72, dtype=np.uint8)
            mask[order[start:start + 6]] = 1
            positive = int(mask[target])
            masks.append(mask)
            labels.append(positive)
            logits.append((3.0 if positive else -1.0) + rng.normal(0, 0.2))
            seqs.append(sequence)
            chars.append(0)
            runs.append("Q_07_SE001_Grey-to-White_Train01")
            subjects.append("Q_07")
            sessions.append("SE001")
    meta = {
        "y": np.asarray(labels), "lit_mask": np.asarray(masks),
        "seq_idx": np.asarray(seqs), "char_idx": np.asarray(chars),
        "run_id": np.asarray(runs), "subject": np.asarray(subjects),
        "session": np.asarray(sessions),
    }
    scores = np.asarray(logits)
    rows = np.arange(len(scores))
    temperature, nll = _fit_temperature(scores, meta, rows)
    assert 0.05 < temperature < 20
    assert np.isfinite(nll)
    metrics, _ = _eval_policy(scores / temperature, meta, rows)
    assert metrics["eeg_only"]["accuracy"] == 1.0
    assert metrics["eeg_only"]["mean_sequences"] <= 4
