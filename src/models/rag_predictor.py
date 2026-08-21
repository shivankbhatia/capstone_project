"""Retrieval-augmented next-character priors for the P300 speller.

The RAG layer is intentionally lightweight: it retrieves prefix-compatible
phrases from a local CSV/JSON phrase bank and converts their continuations into
a probability distribution over the same grid classes used by LLMPredictor.
That retrieval prior is then conservatively interpolated with the base LLM
prior, so existing Bayesian fusion and decoder code can consume it without
knowing whether it came from a plain LLM or a RAG-enhanced predictor.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class RetrievedPhrase:
    """One normalized phrase-bank entry used by the retrieval prior."""

    text: str
    source: str = "phrase_bank"
    weight: float = 1.0
    category: str = "general"


@dataclass(frozen=True)
class RAGDiagnostics:
    """Human-auditable details for the most recent RAG prediction."""

    partial_word: str
    matched_phrases: List[str]
    retrieval_confidence: float
    rag_enabled: bool
    reason: str


class RAGPredictor:
    """Wrap a base character predictor with phrase-bank retrieval.

    Parameters
    ----------
    base_predictor:
        Object exposing ``char_list`` and ``predict_next_char(context_so_far)``.
        ``LLMPredictor`` is the intended production implementation, but tests
        can provide a small deterministic stub.
    phrase_bank_path:
        CSV or JSON file containing text snippets. CSV columns can include
        ``text``, ``source``, ``weight``, and ``category``. JSON can be either a
        list of strings or a list of objects with those fields.
    rag_weight:
        Maximum interpolation weight assigned to retrieval. The effective
        weight is zero when retrieval is unavailable or below the confidence
        threshold.
    retrieval_confidence_threshold:
        Minimum max probability in the retrieval prior required before RAG is
        allowed to influence the base prior.
    max_retrieved_phrases:
        Limits how many compatible phrases contribute to one prediction.
    """

    def __init__(
        self,
        base_predictor,
        phrase_bank_path="data/rag/phrase_bank.csv",
        rag_weight=0.25,
        retrieval_confidence_threshold=0.60,
        max_retrieved_phrases=20,
    ):
        if not 0.0 <= rag_weight <= 1.0:
            raise ValueError("rag_weight must be between 0 and 1")
        if not 0.0 <= retrieval_confidence_threshold <= 1.0:
            raise ValueError("retrieval_confidence_threshold must be between 0 and 1")
        if max_retrieved_phrases <= 0:
            raise ValueError("max_retrieved_phrases must be positive")

        self.base_predictor = base_predictor
        self.char_list = list(base_predictor.char_list)
        self.phrase_bank_path = Path(phrase_bank_path)
        self.rag_weight = rag_weight
        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        self.max_retrieved_phrases = max_retrieved_phrases
        self.phrases = self._load_phrase_bank(self.phrase_bank_path)
        self.last_diagnostics = RAGDiagnostics("", [], 0.0, False, "not_run")

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).replace("_", " ").lower().split())

    @classmethod
    def _coerce_entry(cls, row) -> Optional[RetrievedPhrase]:
        if isinstance(row, str):
            text = cls._normalize_text(row)
            return RetrievedPhrase(text=text) if text else None

        text = cls._normalize_text(row.get("text", ""))
        if not text:
            return None

        raw_weight = row.get("weight", 1.0)
        weight = float(raw_weight) if raw_weight not in (None, "") else 1.0
        return RetrievedPhrase(
            text=text,
            source=str(row.get("source", "phrase_bank") or "phrase_bank"),
            weight=max(weight, 0.0),
            category=str(row.get("category", "general") or "general"),
        )

    @classmethod
    def _load_phrase_bank(cls, path: Path) -> List[RetrievedPhrase]:
        if not path.exists():
            return []

        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
            entries = [cls._coerce_entry(row) for row in rows]
        else:
            with path.open(newline="", encoding="utf-8") as handle:
                entries = [cls._coerce_entry(row) for row in csv.DictReader(handle)]

        return [entry for entry in entries if entry is not None and entry.weight > 0]

    @staticmethod
    def _partial_word(context_so_far: str) -> str:
        clean = RAGPredictor._normalize_text(context_so_far)
        if not clean:
            return ""
        return clean.split(" ")[-1]

    def _char_to_grid_index(self, next_char: str) -> Optional[int]:
        for idx, label in enumerate(self.char_list):
            semantic = " " if label == "Sp" else str(label).lower()
            if semantic == next_char:
                return idx
        return None

    def _matching_phrases(self, partial_word: str) -> List[RetrievedPhrase]:
        if not partial_word:
            return []

        matches = []
        for phrase in self.phrases:
            for token in phrase.text.split():
                if token.startswith(partial_word) and len(token) > len(partial_word):
                    matches.append(phrase)
                    break
            if len(matches) >= self.max_retrieved_phrases:
                break
        return matches

    def retrieval_prior(self, context_so_far: str):
        """Return a retrieval-only prior and diagnostics for the context."""
        partial_word = self._partial_word(context_so_far)
        grid_probs = np.zeros(len(self.char_list), dtype=float)
        matched = self._matching_phrases(partial_word)

        for phrase in matched:
            for token in phrase.text.split():
                if not token.startswith(partial_word) or len(token) <= len(partial_word):
                    continue
                next_char = token[len(partial_word)]
                idx = self._char_to_grid_index(next_char)
                if idx is not None:
                    grid_probs[idx] += phrase.weight

        total = grid_probs.sum()
        if total > 0:
            grid_probs /= total

        confidence = float(grid_probs.max()) if total > 0 else 0.0
        if not matched:
            enabled = False
            reason = "no_matches"
        elif total <= 0:
            enabled = False
            reason = "no_grid_compatible_next_char"
        else:
            enabled = confidence >= self.retrieval_confidence_threshold
            reason = "enabled" if enabled else "below_threshold"

        diagnostics = RAGDiagnostics(
            partial_word=partial_word,
            matched_phrases=[phrase.text for phrase in matched],
            retrieval_confidence=confidence,
            rag_enabled=enabled,
            reason=reason,
        )
        return grid_probs, diagnostics

    @staticmethod
    def _normalize_probs(probs: Iterable[float]) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:
            return np.ones(len(probs), dtype=float) / len(probs)
        return probs / total

    def predict_next_char(self, context_so_far: str):
        """Return a RAG-enhanced grid prior for the next character."""
        base_prior = self._normalize_probs(self.base_predictor.predict_next_char(context_so_far))
        retrieval_prior, diagnostics = self.retrieval_prior(context_so_far)

        if diagnostics.rag_enabled:
            combined = ((1.0 - self.rag_weight) * base_prior) + (self.rag_weight * retrieval_prior)
            prior = self._normalize_probs(combined)
        else:
            prior = base_prior

        self.last_diagnostics = diagnostics
        return prior
