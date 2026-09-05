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
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional

import numpy as np


@dataclass(frozen=True)
class RetrievedPhrase:
    """One normalized phrase-bank entry used by the retrieval prior."""

    text: str
    source: str = "phrase_bank"
    weight: float = 1.0
    category: str = "general"


@dataclass(frozen=True)
class RetrievalMatch:
    """One phrase continuation candidate for a normalized context suffix."""

    phrase: RetrievedPhrase
    matched_context: str
    next_char: str
    score: float
    phrase_prefix_match: bool


@dataclass(frozen=True)
class RAGDiagnostics:
    """Human-auditable details for the most recent RAG prediction."""

    partial_word: str
    matched_phrases: List[str]
    retrieval_confidence: float
    rag_enabled: bool
    reason: str
    normalized_context: str = ""
    matched_context: str = ""
    top_margin: float = 0.0
    subject_weight: float = 0.0
    personalization_active: bool = False
    subject_source: Literal["word_match", "char_backoff", "none"] = "none"
    subject_gate_weight: float = 0.0
    global_gate_weight: float = 0.0


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
        subject_id: Optional[str] = None,
        min_subject_phrases: int = 10,
        personalization_bonus: float = 1.5,
        subject_only: bool = False,
        subject_confidence_threshold: float = 0.20,
    ):
        if not 0.0 <= rag_weight <= 1.0:
            raise ValueError("rag_weight must be between 0 and 1")
        if not 0.0 <= retrieval_confidence_threshold <= 1.0:
            raise ValueError(
                "retrieval_confidence_threshold must be between 0 and 1"
            )
        if max_retrieved_phrases <= 0:
            raise ValueError("max_retrieved_phrases must be positive")
        if min_subject_phrases <= 0:
            raise ValueError("min_subject_phrases must be positive")
        if personalization_bonus < 0:
            raise ValueError("personalization_bonus must be non-negative")
        if not 0.0 <= subject_confidence_threshold <= 1.0:
            raise ValueError("subject_confidence_threshold must be between 0 and 1")

        self.base_predictor = base_predictor
        self.char_list = list(base_predictor.char_list)
        self.phrase_bank_path = Path(phrase_bank_path)
        self.rag_weight = rag_weight
        self.retrieval_confidence_threshold = retrieval_confidence_threshold
        self.max_retrieved_phrases = max_retrieved_phrases
        self.subject_id = subject_id
        self.min_subject_phrases = min_subject_phrases
        self.personalization_bonus = personalization_bonus
        self.subject_only = subject_only
        self.subject_confidence_threshold = subject_confidence_threshold
        self.global_phrases = self._load_phrase_bank(self.phrase_bank_path)
        self.subject_phrase_bank_path = self._subject_phrase_bank_path()
        self.subject_phrases = (
            self._load_phrase_bank(self.subject_phrase_bank_path)
            if self.subject_phrase_bank_path is not None
            else []
        )
        # Backwards-compatible alias for callers which inspect the original
        # single phrase list. Retrieval itself uses the two banks separately.
        self.phrases = self.global_phrases
        self.last_diagnostics = RAGDiagnostics("", [], 0.0, False, "not_run")

    def _subject_phrase_bank_path(self) -> Optional[Path]:
        if not self.subject_id:
            return None
        return (
            self.phrase_bank_path.parent
            / "by_subject"
            / f"phrase_bank_{self.subject_id}.csv"
        )

    @property
    def subject_weight(self) -> float:
        """Continuous offline-plus-online personalization strength."""
        if not self.subject_id:
            return 0.0
        return min(1.0, len(self.subject_phrases) / self.min_subject_phrases)

    def add_observed_phrase(self, text: str) -> bool:
        """Add a completed phrase to this predictor's in-memory subject bank.

        The method deliberately does not write to disk: evaluation callers can
        use it for within-session adaptation without leaking held-out targets
        into a future evaluation session.
        """
        if not self.subject_id:
            return False
        phrase = self._coerce_entry(
            {
                "text": text,
                "source": "observed_subject_session",
                "weight": 1.0,
                "category": "observed",
            }
        )
        if phrase is None or phrase.weight <= 0:
            return False
        self.subject_phrases.append(phrase)
        return True

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).replace("_", " ").lower().split())

    @staticmethod
    def _normalize_context(context: str) -> str:
        raw = str(context).replace("_", " ").lower()
        normalized = " ".join(raw.split())

        if normalized and raw[-1:].isspace():
            return f"{normalized} "

        return normalized

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
                entries = [
                    cls._coerce_entry(row)
                    for row in csv.DictReader(handle)
                ]

        return [
            entry
            for entry in entries
            if entry is not None and entry.weight > 0
        ]

    @classmethod
    def _partial_word(cls, context_so_far: str) -> str:
        clean = cls._normalize_context(context_so_far).rstrip()

        if not clean:
            return ""

        return clean.split(" ")[-1]

    @staticmethod
    def _context_suffixes(normalized_context: str) -> List[str]:
        if not normalized_context:
            return []

        stripped = normalized_context.rstrip()
        suffixes = [normalized_context]

        start = 0

        while True:
            start = stripped.find(" ", start)

            if start == -1:
                break

            suffix = normalized_context[start + 1 :]

            if suffix and suffix not in suffixes:
                suffixes.append(suffix)

            start += 1

        return sorted(suffixes, key=len, reverse=True)

    def _char_to_grid_index(self, next_char: str) -> Optional[int]:
        for idx, label in enumerate(self.char_list):
            semantic = " " if label == "Sp" else str(label).lower()

            if semantic == next_char:
                return idx

        return None

    def _matching_phrases(
        self,
        context_so_far: str,
        phrases: Optional[List[RetrievedPhrase]] = None,
    ) -> List[RetrievalMatch]:
        normalized_context = self._normalize_context(context_so_far)
        suffixes = self._context_suffixes(normalized_context)

        if not suffixes:
            return []

        matches: List[RetrievalMatch] = []

        if phrases is None:
            phrases = self.subject_phrases if self.subject_only else self.global_phrases

        for phrase in phrases:
            best_match: Optional[RetrievalMatch] = None

            for suffix in suffixes:
                for start in self._phrase_match_starts(phrase.text, suffix):
                        end = start + len(suffix)

                        if end >= len(phrase.text):
                            continue

                        next_char = phrase.text[end]

                        if self._char_to_grid_index(next_char) is None:
                            continue

                        prefix_match = (
                            start == 0
                            and suffix == normalized_context
                        )

                        context_bonus = 2.0 if prefix_match else 1.0

                        score = (
                            phrase.weight
                            * (1.0 + len(suffix))
                            * context_bonus
                        )

                        candidate = RetrievalMatch(
                            phrase=phrase,
                            matched_context=suffix,
                            next_char=next_char,
                            score=score,
                            phrase_prefix_match=prefix_match,
                        )

                        if (
                            best_match is None
                            or candidate.score > best_match.score
                        ):
                            best_match = candidate

            if best_match is not None:
                matches.append(best_match)

        matches.sort(
            key=lambda match: (
                len(match.matched_context),
                match.score,
            ),
            reverse=True,
        )

        return matches[: self.max_retrieved_phrases]

    @staticmethod
    def _phrase_match_starts(
        phrase_text: str,
        suffix: str,
    ) -> List[int]:
        starts = []
        search_from = 0

        while True:
            start = phrase_text.find(suffix, search_from)

            if start == -1:
                return starts

            at_word_boundary = (
                start == 0
                or phrase_text[start - 1] == " "
            )

            if at_word_boundary:
                starts.append(start)

            search_from = start + 1

    @staticmethod
    def _soft_gate(confidence, threshold, sharpness=10.0):
        return 1.0 / (1.0 + math.exp(-sharpness * (confidence - threshold)))

    def _prior_from_matches(self, matches):
        grid_probs = np.zeros(len(self.char_list), dtype=float)

        for match in matches:
            idx = self._char_to_grid_index(match.next_char)

            if idx is not None:
                grid_probs[idx] += match.score

        total = grid_probs.sum()

        if total > 0:
            grid_probs /= total

        confidence = (
            float(grid_probs.max())
            if total > 0
            else 0.0
        )

        sorted_probs = np.sort(grid_probs)

        top_margin = (
            float(sorted_probs[-1] - sorted_probs[-2])
            if len(sorted_probs) > 1
            else confidence
        )

        if not matches:
            reason = "no_matches"
        elif total <= 0:
            reason = "no_grid_compatible_next_char"
        else:
            reason = "available"
        return grid_probs, confidence, reason

    def _char_ngram_prior(self, context_so_far, phrases):
        """Order-2, Laplace-smoothed character backoff over subject text."""
        text = " ".join(phrase.text for phrase in phrases)
        bigram_counts, unigram_counts = Counter(), Counter()
        for first, second in zip(text, text[1:]):
            bigram_counts[(first, second)] += 1
            unigram_counts[first] += 1
        context = self._normalize_context(context_so_far)
        last = context[-1] if context else " "
        probs = np.ones(len(self.char_list), dtype=float)
        for index, label in enumerate(self.char_list):
            char = " " if label == "Sp" else str(label).lower()
            probs[index] += bigram_counts.get((last, char), 0)
        probs /= probs.sum()
        confidence = float((probs.max() * len(probs) - 1) / max(1, len(probs) - 1))
        return probs, confidence

    def _bank_prior(self, context_so_far, phrases):
        matches = self._matching_phrases(context_so_far, phrases)
        prior, confidence, reason = self._prior_from_matches(matches)
        return prior, confidence, reason, matches

    def retrieval_prior(self, context_so_far: str):
        """Return the weighted retrieval prior and diagnostics for a context."""
        normalized_context = self._normalize_context(context_so_far)
        partial_word = self._partial_word(context_so_far)
        global_prior, global_confidence, global_reason, global_matches = self._bank_prior(
            context_so_far, self.global_phrases
        )
        subject_prior, subject_confidence, subject_reason, subject_matches = self._bank_prior(
            context_so_far, self.subject_phrases
        )
        subject_source = "word_match" if subject_matches else "none"
        if subject_source == "none" and self.subject_phrases:
            subject_prior, subject_confidence = self._char_ngram_prior(
                context_so_far, self.subject_phrases
            )
            subject_source = "char_backoff"

        global_gate = (
            self._soft_gate(global_confidence, self.retrieval_confidence_threshold, 30.0)
            if global_reason == "available" and not self.subject_only else 0.0
        )
        subject_gate = (
            self._soft_gate(subject_confidence, self.subject_confidence_threshold, 8.0)
            if subject_source != "none" else 0.0
        )
        global_weight = self.rag_weight * global_gate
        subject_gate_weight = (
            self.rag_weight * self.personalization_bonus * self.subject_weight * subject_gate
        )
        total_weight = global_weight + subject_gate_weight
        if total_weight > 0.9:
            scale = 0.9 / total_weight
            global_weight *= scale
            subject_gate_weight *= scale
        weighted_retrieval = self._normalize_probs(
            global_prior * global_weight + subject_prior * subject_gate_weight
        )
        all_matches = global_matches + subject_matches
        confidence = max(global_confidence, subject_confidence)
        top_margin = float(np.sort(weighted_retrieval)[-1] - np.sort(weighted_retrieval)[-2])
        active = global_weight > 0 or subject_gate_weight > 0

        diagnostics = RAGDiagnostics(
            partial_word=partial_word,
            matched_phrases=[match.phrase.text for match in all_matches],
            retrieval_confidence=confidence,
            rag_enabled=active,
            reason="enabled" if active else (subject_reason if self.subject_id else global_reason),
            normalized_context=normalized_context,
            matched_context=all_matches[0].matched_context if all_matches else "",
            top_margin=top_margin,
            subject_weight=self.subject_weight,
            personalization_active=subject_gate_weight > 0,
            subject_source=subject_source,
            subject_gate_weight=subject_gate_weight,
            global_gate_weight=global_weight,
        )
        return weighted_retrieval, diagnostics

    @staticmethod
    def _normalize_probs(probs: Iterable[float]) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.clip(probs, 0.0, None)

        total = probs.sum()

        if total <= 0:
            return (
                np.ones(len(probs), dtype=float)
                / len(probs)
            )

        return probs / total

    def predict_next_char(self, context_so_far: str):
        """Return a RAG-enhanced grid prior for the next character."""

        base_prior = self._normalize_probs(
            self.base_predictor.predict_next_char(
                context_so_far
            )
        )

        retrieval_prior, diagnostics = self.retrieval_prior(
            context_so_far
        )

        retrieval_weight = diagnostics.subject_gate_weight + diagnostics.global_gate_weight
        prior = self._normalize_probs(
            (1.0 - retrieval_weight) * base_prior + retrieval_weight * retrieval_prior
        )

        self.last_diagnostics = diagnostics

        return prior
