from dataclasses import fields
import inspect

import numpy as np
import pytest

from src.stimulus.scheduler import (
    SCHEDULER_MODES,
    LanguageGuidedScheduler,
    SchedulerState,
)


def test_scheduler_api_has_no_target_input_or_state_field():
    assert "target" not in {field.name for field in fields(SchedulerState)}
    assert list(inspect.signature(LanguageGuidedScheduler.next_flash).parameters) == ["self", "state"]
    scheduler = LanguageGuidedScheduler()
    with pytest.raises(TypeError, match="SchedulerState"):
        scheduler.next_flash({"posterior": [1 / 72] * 72, "target": "A"})


@pytest.mark.parametrize("mode", SCHEDULER_MODES)
def test_each_scheduler_mode_returns_unflashed_code(mode):
    posterior = tuple([1 / 72] * 72)
    prior = np.zeros(72)
    prior[0] = 1.0
    state = SchedulerState(
        posterior=posterior,
        flashed_codes=(1,),
        lm_prior=tuple(prior),
        rag_prior=tuple(prior),
    )
    code = LanguageGuidedScheduler(mode=mode, seed=4).next_flash(state)
    assert 1 <= code <= 17
    assert code != 1


def test_scheduler_stops_on_observed_confidence_after_minimum_flashes():
    posterior = np.full(72, 0.1 / 71)
    posterior[0] = 0.9
    state = SchedulerState(posterior=tuple(posterior), flashed_codes=(1, 10))
    scheduler = LanguageGuidedScheduler(confidence_stop=0.8, min_flashes=2)
    assert scheduler.next_flash(state) is None


def test_scheduler_accepts_target_blind_group_masks_for_study_q():
    masks = np.zeros((3, 72), dtype=int)
    masks[0, :6] = 1
    masks[1, 6:12] = 1
    masks[2, 12:18] = 1
    prior = np.zeros(72)
    prior[0] = 1.0
    state = SchedulerState(
        posterior=tuple(np.full(72, 1 / 72)),
        lm_prior=tuple(prior), rag_prior=tuple(prior),
        stimulus_masks=tuple(tuple(int(v) for v in row) for row in masks),
    )
    scheduler = LanguageGuidedScheduler(mode="lm_guided", exploration_floor=0.0, seed=4)
    assert scheduler.next_flash(state) == 1
    assert "target" not in {field.name for field in fields(SchedulerState)}
