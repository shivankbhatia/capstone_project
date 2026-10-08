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
