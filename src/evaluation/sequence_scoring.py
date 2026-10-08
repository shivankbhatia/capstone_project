"""Shared row/column scoring, sequential stopping, and variable-time ITR.

All evaluator paths use these primitives so changes to the decoder policy are
exercised consistently on OOF data and held-out replays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class SequenceEvidence:
    row: np.ndarray
    column: np.ndarray
    grid: np.ndarray
    valid_flash_count: int


def aggregate_flash_logits(logits, stimulus_codes, n_rows=9, n_cols=8):
    """Sum per-flash logits by row/column and return calibrated posteriors."""
    logits = np.asarray(logits, dtype=float).reshape(-1)
    codes = np.asarray(stimulus_codes, dtype=int).reshape(-1)
    if logits.shape != codes.shape:
        raise ValueError("logits and stimulus_codes must have the same shape")
    row_logits = np.zeros(n_rows, dtype=float)
    col_logits = np.zeros(n_cols, dtype=float)
    valid = np.isfinite(logits) & (codes >= 1) & (codes <= n_rows + n_cols)
    for value, code in zip(logits[valid], codes[valid]):
        if code <= n_rows:
            row_logits[code - 1] += value
        else:
            col_logits[code - n_rows - 1] += value
    has_rows = np.any(valid & (codes <= n_rows))
    has_cols = np.any(valid & (codes > n_rows))
    if not (has_rows and has_cols):
        return None
    row = softmax(row_logits)
    column = softmax(col_logits)
    return SequenceEvidence(row, column, np.outer(row, column).ravel(), int(valid.sum()))


def aggregate_membership_logits(logits, lit_masks, n_rows=9, n_cols=8):
    """Aggregate group-flash logits into a grid posterior.

    ``lit_masks`` contains one row-major binary membership vector per flash.
    Each cell accumulates the target log-odds from every flash that included
    it, matching the sufficient statistic used by the group-flash decoder.
    """
    values = np.asarray(logits, dtype=float).reshape(-1)
    masks = np.asarray(lit_masks)
    n_cells = int(n_rows) * int(n_cols)
    if masks.ndim == 1:
        # Accept the parser's compact strings as well as numeric matrices.
        if len(masks) and isinstance(masks[0], (str, bytes)):
            masks = np.asarray([[int(bit) for bit in mask] for mask in masks], dtype=float)
        else:
            raise ValueError("lit_masks must be a 2-D membership matrix or binary strings")
    if masks.shape != (len(values), n_cells):
        raise ValueError(f"lit_masks must have shape ({len(values)}, {n_cells})")
    if not np.isfinite(values).all() or not np.isfinite(masks).all():
        raise ValueError("logits and lit_masks must be finite")
    if not np.isin(masks, (0, 1)).all():
        raise ValueError("lit_masks must contain only zero and one")
    scores = masks.T @ values
    return softmax(scores)


class SequentialDecoder:
    """Accumulate per-sequence grid posteriors under a single stopping rule."""

    def __init__(self, n_classes, initial_log_bias=None, tau=0.80,
                 min_sequences=2, max_sequences=10):
        if n_classes < 2:
            raise ValueError("n_classes must be at least 2")
        if not (0.0 <= tau <= 1.0):
            raise ValueError("tau must be between 0 and 1")
        if min_sequences < 1 or max_sequences < min_sequences:
            raise ValueError("Require 1 <= min_sequences <= max_sequences")
        bias = np.zeros(n_classes) if initial_log_bias is None else np.asarray(initial_log_bias, dtype=float)
        if bias.shape != (n_classes,) or not np.isfinite(bias).all():
            raise ValueError("initial_log_bias must be a finite vector of length n_classes")
        self.n_classes = int(n_classes)
        self.initial_log_bias = bias.copy()
        self.log_posterior = bias.copy()
        self.tau = float(tau)
        self.min_sequences = int(min_sequences)
        self.max_sequences = int(max_sequences)
        self.sequences_used = 0
        self.stopped = False

    @property
    def posterior(self):
        return softmax(self.log_posterior)

    def add(self, grid_posterior):
        """Add one valid grid posterior; return current posterior and stop flag."""
        if self.stopped or self.sequences_used >= self.max_sequences:
            return self.posterior, self.stopped
        probs = np.asarray(grid_posterior, dtype=float)
        if probs.shape != (self.n_classes,) or not np.isfinite(probs).all():
            raise ValueError("grid_posterior must be a finite vector of length n_classes")
        if np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError("grid_posterior must contain nonnegative probability mass")
        probs = probs / probs.sum()
        self.log_posterior += np.log(np.clip(probs, 1e-9, 1.0))
        self.sequences_used += 1
        post = self.posterior
        self.stopped = self.sequences_used >= self.min_sequences and float(post.max()) >= self.tau
        return post, self.stopped

    def result(self):
        post = self.posterior
        return {
            "decision_idx": int(post.argmax()) if self.sequences_used else None,
            "confidence": float(post.max()) if self.sequences_used else None,
            "sequences_used": self.sequences_used,
            "stopped_by_threshold": self.stopped,
            "posterior": post,
        }


def variable_time_itr(num_classes, accuracy, seconds_per_character):
    """Wolpaw ITR in bits/min using observed mean seconds per character."""
    if seconds_per_character <= 0:
        raise ValueError("seconds_per_character must be > 0")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    p = float(accuracy)
    bits = np.log2(num_classes)
    if p > 0:
        bits += p * np.log2(p)
    if p < 1:
        bits += (1 - p) * np.log2((1 - p) / (num_classes - 1))
    return float(max(0.0, bits) * 60.0 / float(seconds_per_character))
