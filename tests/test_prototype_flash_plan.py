import numpy as np

from prototype.server import ReplayService


def test_first_sequence_is_row_then_column_and_exposes_classifier_score():
    service = ReplayService()
    prior = np.full(len(service.char_list), 1 / len(service.char_list))

    plan = service._flash_plan(
        [
            {"stimulus_code": 10, "target_probability": 0.12},
            {"stimulus_code": 1, "target_probability": 0.81},
            {"stimulus_code": 11, "target_probability": 0.09},
        ],
        prior,
        sequence=1,
    )

    assert [flash["label"] for flash in plan] == ["ROW 1", "COLUMN 1", "COLUMN 2"]
    assert plan[0]["keys"] == list("ABCDEFGH")
    assert plan[0]["classifier_target"] is True


def test_later_sequence_orders_groups_by_language_prior_mass():
    service = ReplayService()
    prior = np.zeros(len(service.char_list))
    prior[service.char_list.index("F")] = 1.0  # F is in row 1 and column 6.

    plan = service._flash_plan(
        [
            {"stimulus_code": 11, "target_probability": 0.2},  # column 2
            {"stimulus_code": 1, "target_probability": 0.2},   # row 1
            {"stimulus_code": 15, "target_probability": 0.2},  # column 6
        ],
        prior,
        sequence=2,
    )

    assert plan[0]["label"] == "ROW 1"
    assert plan[1]["label"] == "COLUMN 6"
