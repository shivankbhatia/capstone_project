"""Target-blind schedulers for row/column P300 flashing.

The API accepts only observed probabilities and presentation history. Ground-
truth targets are deliberately absent from the state type and method signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


SCHEDULER_MODES = ("uniform", "lm_guided", "lm_rag", "closed_loop")


@dataclass(frozen=True)
class SchedulerState:
    """Observable state available online; it intentionally has no target field."""

    posterior: Tuple[float, ...]
    flashed_codes: Tuple[int, ...] = ()
    lm_prior: Optional[Tuple[float, ...]] = None
    rag_prior: Optional[Tuple[float, ...]] = None
    n_rows: int = 9
    n_cols: int = 8
    stimulus_masks: Tuple[Tuple[int, ...], ...] = ()


class LanguageGuidedScheduler:
    """Select the next row/column code without access to ground truth."""

    def __init__(self, mode="uniform", exploration_floor=0.20,
                 confidence_stop=0.80, min_flashes=2, seed=0):
        if mode not in SCHEDULER_MODES:
            raise ValueError(f"mode must be one of {SCHEDULER_MODES}")
        if not 0.0 <= exploration_floor <= 1.0:
            raise ValueError("exploration_floor must be between 0 and 1")
        if not 0.0 <= confidence_stop <= 1.0:
            raise ValueError("confidence_stop must be between 0 and 1")
        if min_flashes < 1:
            raise ValueError("min_flashes must be positive")
        self.mode = mode
        self.exploration_floor = float(exploration_floor)
        self.confidence_stop = float(confidence_stop)
        self.min_flashes = int(min_flashes)
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def _distribution(values, n_classes):
        if values is None:
            return np.full(n_classes, 1.0 / n_classes)
        probs = np.asarray(values, dtype=float)
        if probs.shape != (n_classes,) or not np.isfinite(probs).all() or np.any(probs < 0):
            raise ValueError(f"prior/posterior must be a finite nonnegative vector of length {n_classes}")
        total = probs.sum()
        return probs / total if total > 0 else np.full(n_classes, 1.0 / n_classes)

    @staticmethod
    def _group_masses(distribution, n_rows, n_cols):
        if distribution.size != n_rows * n_cols:
            raise ValueError("probability-vector length must equal n_rows * n_cols")
        grid = distribution.reshape(n_rows, n_cols)
        return np.concatenate((grid.sum(axis=1), grid.sum(axis=0)))

    def _weights(self, state):
        n_classes = state.n_rows * state.n_cols
        if state.stimulus_masks:
            masks = np.asarray(state.stimulus_masks, dtype=float)
            if masks.ndim != 2 or masks.shape[1] != n_classes or not np.isin(masks, (0, 1)).all():
                raise ValueError("stimulus_masks must be a binary (n_stimuli, n_classes) matrix")
            if np.any(masks.sum(axis=1) == 0):
                raise ValueError("stimulus_masks cannot contain empty groups")
            group_masses = lambda distribution: masks @ distribution
            if self.mode == "uniform":
                return np.ones(len(masks), dtype=float)
            lm = self._distribution(state.lm_prior, n_classes)
            if self.mode == "lm_guided":
                return group_masses(lm)
            rag = self._distribution(state.rag_prior, n_classes)
            if self.mode == "lm_rag":
                return 0.5 * (group_masses(lm) + group_masses(rag))
            posterior = self._distribution(state.posterior, n_classes)
            closed = self._distribution(posterior * lm * rag, n_classes)
            return group_masses(closed)
        if self.mode == "uniform":
            return np.ones(state.n_rows + state.n_cols, dtype=float)
        lm = self._distribution(state.lm_prior, n_classes)
        lm_masses = self._group_masses(lm, state.n_rows, state.n_cols)
        if self.mode == "lm_guided":
            return lm_masses
        rag = self._distribution(state.rag_prior, n_classes)
        rag_masses = self._group_masses(rag, state.n_rows, state.n_cols)
        if self.mode == "lm_rag":
            return 0.5 * (lm_masses + rag_masses)
        posterior = self._distribution(state.posterior, n_classes)
        closed = self._distribution(posterior * lm * rag, n_classes)
        return self._group_masses(closed, state.n_rows, state.n_cols)

    def next_flash(self, state: SchedulerState):
        """Return a 1-based row/column stimulus code, or None when confidence stops."""
        if not isinstance(state, SchedulerState):
            raise TypeError("state must be SchedulerState; targets and arbitrary mappings are not accepted")
        posterior = self._distribution(state.posterior, state.n_rows * state.n_cols)
        if len(state.flashed_codes) >= self.min_flashes and posterior.max() >= self.confidence_stop:
            return None
        n_stimuli = len(state.stimulus_masks) if state.stimulus_masks else state.n_rows + state.n_cols
        all_codes = np.arange(1, n_stimuli + 1)
        already = set(int(code) for code in state.flashed_codes)
        if any(code < 1 or code > len(all_codes) for code in already):
            raise ValueError("flashed_codes contains an invalid row/column code")
        candidates = np.asarray([code for code in all_codes if code not in already], dtype=int)
        if not len(candidates):
            candidates = all_codes
        weights = np.asarray(self._weights(state), dtype=float)[candidates - 1]
        weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
        floor = self.exploration_floor / len(candidates)
        probabilities = floor + (1.0 - self.exploration_floor) * weights
        probabilities /= probabilities.sum()
        return int(self._rng.choice(candidates, p=probabilities))
