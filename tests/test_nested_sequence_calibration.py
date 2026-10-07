import numpy as np

from scripts.nested_sequence_calibration import _sequence_metrics, _sequence_nll


def test_sequence_metrics_use_row_and_column_softmax_targets():
    codes = np.arange(1, 18, dtype=int)
    y = ((codes == 4) | (codes == 11)).astype(np.uint8)
    logits = np.where(y == 1, 2.0, -0.5)
    metadata = (
        np.asarray(["D_01_SE001_RC_Train01"] * 17),
        np.zeros(17, dtype=int),
        np.ones(17, dtype=int),
        codes,
    )
    metrics = _sequence_metrics(logits, y, metadata)
    assert metrics["complete_sequences"] == 1
    assert metrics["row_accuracy"] == 1.0
    assert metrics["column_accuracy"] == 1.0
    assert metrics["row_column_nll"] < np.log(8.5)
    assert np.isclose(_sequence_nll(0.0, logits, y, metadata), metrics["row_column_nll"])
