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
    # The three factors below make the final subject contribution auditable:
    # confidence sigmoid, cold-start ramp, and their final blend weight.
    subject_confidence_weight: float = 0.0
    subject_coldstart_weight: float = 0.0
    subject_blend_weight: float = 0.0
    subject_gate_weight: float = 0.0
    global_gate_weight: float = 0.0
    sufficiency_gate_value: float = 0.0
    subject_token_count: int = 0


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
        Retrieval is blended continuously with the base prior.  The configured
        threshold is the midpoint of a sigmoid, not a binary eligibility gate.
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
        # Study-D subject banks contain only 12--17 words.  Centre the
        # existing cold-start ramp in that observed range so it attenuates
        # sparse evidence instead of reducing every subject contribution to
        # a numerically irrelevant value.
        sufficiency_midpoint_tokens: float = 15.0,
        sufficiency_sharpness: float = 0.20,
        subject_gate_sharpness: float = 10.0,
        rung_id: Optional[str] = None,
        disable_bigram_backoff: bool = False,
        bigram_min_count: int = 2,
        bigram_max_normalized_entropy: float = 0.65,
        debug_trace_calls: int = 0,
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
        if sufficiency_midpoint_tokens <= 0:
            raise ValueError("sufficiency_midpoint_tokens must be positive")
        if sufficiency_sharpness <= 0:
            raise ValueError("sufficiency_sharpness must be positive")
        if subject_gate_sharpness <= 0:
            raise ValueError("subject_gate_sharpness must be positive")
        if bigram_min_count < 1:
            raise ValueError("bigram_min_count must be at least one")
        if not 0.0 <= bigram_max_normalized_entropy <= 1.0:
            raise ValueError("bigram_max_normalized_entropy must be between 0 and 1")

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
        self.sufficiency_midpoint_tokens = sufficiency_midpoint_tokens
        self.sufficiency_sharpness = sufficiency_sharpness
        self.subject_gate_sharpness = subject_gate_sharpness
        self.rung_id = rung_id or "unspecified"
        self.disable_bigram_backoff = disable_bigram_backoff
        self.bigram_min_count = bigram_min_count
        self.bigram_max_normalized_entropy = bigram_max_normalized_entropy
        self.debug_trace_calls = debug_trace_calls
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
        # Total word count backing the subject bank -- the raw signal the
        # data-sufficiency gate trusts.
        self.subject_token_count = sum(
            len(phrase.text.split()) for phrase in self.subject_phrases
        )
        self._subject_bigrams, self._subject_unigrams = self._build_char_ngram_table(
            self.subject_phrases
        )
        # These are deliberately in-memory diagnostics: callers can inspect
        # every character decision without changing evaluation outcomes.
        self.char_backoff_count = 0
        self.source_counts = {
            "word_level_hits": 0,
            "bigram_backoff": 0,
            "below_min_count": 0,
            "high_entropy": 0,
        }
        self.flip_counts = {"total": 0, "toward_target": 0, "away_from_target": 0}
        self.blend_history = []

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
        self.subject_token_count += len(phrase.text.split())
        self._add_phrase_to_char_ngram_table(phrase)
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

    def _data_sufficiency_gate(self) -> float:
        """How much to trust the subject bank based on its corpus size alone.

        Independent of any single query's match confidence -- a subject
        bank of 15 words stays near-zero here regardless of how confident
        an individual retrieval looks, which is what prevents the rung-3
        regression (sparse-bank noise masquerading as high confidence).
        """
        return self._soft_gate(
            float(self.subject_token_count),
            self.sufficiency_midpoint_tokens,
            self.sufficiency_sharpness,
        )

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

    @staticmethod
    def _build_char_ngram_table(phrases):
        """Build one per-subject bigram table, excluding synthetic joins."""
        bigram_counts, unigram_counts = Counter(), Counter()
        for phrase in phrases:
            for first, second in zip(phrase.text, phrase.text[1:]):
                bigram_counts[(first, second)] += phrase.weight
                unigram_counts[first] += phrase.weight
        return bigram_counts, unigram_counts

    def _add_phrase_to_char_ngram_table(self, phrase):
        for first, second in zip(phrase.text, phrase.text[1:]):
            self._subject_bigrams[(first, second)] += phrase.weight
            self._subject_unigrams[first] += phrase.weight

    def _char_ngram_prior(self, context_so_far):
        """Return a gated empirical bigram prior, or ``None`` to use base LM."""
        context = self._normalize_context(context_so_far)
        last = context[-1] if context else " "
        supported = []
        for index, label in enumerate(self.char_list):
            char = " " if label == "Sp" else str(label).lower()
            count = self._subject_bigrams.get((last, char), 0)
            if count >= self.bigram_min_count:
                supported.append((index, count))

        if not supported:
            self.source_counts["below_min_count"] += 1
            return None, 0.0

        counts = np.asarray([count for _, count in supported], dtype=float)
        probabilities = counts / counts.sum()
        # Normalize by the entropy of a uniform distribution over the actual
        # supported successors: 0=single unambiguous successor, 1=uniform.
        entropy = (
            float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))
            if len(counts) > 1 else 0.0
        )
        if entropy > self.bigram_max_normalized_entropy:
            self.source_counts["high_entropy"] += 1
            return None, 0.0

        probs = np.zeros(len(self.char_list), dtype=float)
        for (index, _), probability in zip(supported, probabilities):
            probs[index] = probability
        probs /= probs.sum()
        confidence = float(probs.max())
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
        if subject_source == "word_match":
            self.source_counts["word_level_hits"] += 1
        elif (
            subject_source == "none"
            and self.subject_phrases
            and not self.disable_bigram_backoff
        ):
            subject_prior, subject_confidence = self._char_ngram_prior(context_so_far)
            if subject_prior is not None:
                subject_source = "char_backoff"
                self.char_backoff_count += 1
                self.source_counts["bigram_backoff"] += 1
            else:
                subject_prior = np.zeros(len(self.char_list), dtype=float)

        global_gate = (
            self._soft_gate(global_confidence, self.retrieval_confidence_threshold, 30.0)
            if global_reason == "available" and not self.subject_only else 0.0
        )
        subject_gate = (
            self._soft_gate(
                subject_confidence,
                self.subject_confidence_threshold,
                self.subject_gate_sharpness,
            )
            if subject_source != "none" else 0.0
        )
        sufficiency_gate = self._data_sufficiency_gate()
        global_weight = self.rag_weight * global_gate
        subject_gate_weight = (
            self.rag_weight * self.personalization_bonus * self.subject_weight
            * subject_gate * sufficiency_gate
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
            subject_confidence_weight=subject_gate,
            subject_coldstart_weight=sufficiency_gate,
            subject_blend_weight=subject_gate_weight,
            subject_gate_weight=subject_gate_weight,
            global_gate_weight=global_weight,
            sufficiency_gate_value=sufficiency_gate,
            subject_token_count=self.subject_token_count,
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

    def predict_next_char(self, context_so_far: str, target_char: Optional[str] = None):
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

        base_argmax = int(np.argmax(base_prior))
        final_argmax = int(np.argmax(prior))
        flipped = base_argmax != final_argmax
        flip_direction = "none"
        if flipped:
            self.flip_counts["total"] += 1
            if target_char is not None:
                target_index = self._char_to_grid_index(str(target_char).lower())
                if final_argmax == target_index:
                    flip_direction = "toward_target"
                    self.flip_counts["toward_target"] += 1
                elif base_argmax == target_index:
                    flip_direction = "away_from_target"
                    self.flip_counts["away_from_target"] += 1
                else:
                    flip_direction = "other"

        self.last_diagnostics = diagnostics
        self.blend_history.append({
            "context": str(context_so_far),
            "subject_source": diagnostics.subject_source,
            "confidence": diagnostics.retrieval_confidence,
            "confidence_weight": diagnostics.subject_confidence_weight,
            "coldstart_weight": diagnostics.subject_coldstart_weight,
            "blend_weight": diagnostics.subject_blend_weight,
            "base_argmax": base_argmax,
            "final_argmax": final_argmax,
            "flipped": flipped,
            "flip_direction": flip_direction,
        })
        if len(self.blend_history) <= self.debug_trace_calls:
            print(
                f"RAG_TRACE rung={self.rung_id} w={retrieval_weight:.8f} "
                f"rag={np.array2string(retrieval_prior[:5], precision=6)} "
                f"base={np.array2string(base_prior[:5], precision=6)}"
            )

        return prior
