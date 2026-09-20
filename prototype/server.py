#!/usr/bin/env python3
"""Local API and static server for the Neural Type prototype.

The server has two deliberately distinct execution paths:

* ``study_d_replay`` replays a processed Study-D session through the saved
  SWLDA classifier, RAG prior, Bayesian fusion engine, and P300 decoder.
* ``custom_simulation`` uses the same grid, language prior, fusion engine, and
  decoder, but generates target-conditioned EEG likelihoods.  Custom text has
  no recorded EEG, so this is labelled as a simulation in every response.

Run from the repository root:

    .venv/bin/python prototype/server.py

Then open http://127.0.0.1:8000.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import sys
from dataclasses import asdict
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE_DIR = ROOT / "prototype"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GRID_PATH = ROOT / "data/processed/grid_layout.json"
REGISTRY_PATH = ROOT / "data/processed/ground_truth_registry.json"
TEST_SPLIT_PATH = ROOT / "data/processed/test_sessions.json"
MODEL_PATH = ROOT / "data/processed/swlda_model.pkl"
GLOBAL_BANK_PATH = ROOT / "data/rag/phrase_bank_global.csv"


class ReplayService:
    """Thin adapter around the repository's evaluation components."""

    def __init__(self) -> None:
        self.matrix, self.n_rows, self.n_cols = self._load_matrix()
        self.char_list = list(self.matrix.flatten())
        self._classifier = None
        self._predictor = None
        self._predictor_error = None
        self._replay_cache: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _load_matrix():
        from run_pipeline import load_spelling_matrix

        return load_spelling_matrix(str(GRID_PATH), study_name="StudyD")

    def _load_classifier(self):
        if self._classifier is None:
            self._classifier = joblib.load(MODEL_PATH)
        return self._classifier

    def _load_predictor(self):
        """Load the real DistilGPT-2 + global RAG predictor once, lazily."""
        if self._predictor is not None:
            return self._predictor
        if self._predictor_error is not None:
            raise RuntimeError(self._predictor_error)

        try:
            from src.models.llm_predictor import LLMPredictor
            from src.models.rag_predictor import RAGPredictor

            base = LLMPredictor(self.matrix)
            self._predictor = RAGPredictor(
                base,
                phrase_bank_path=str(GLOBAL_BANK_PATH),
                rag_weight=0.15,
                retrieval_confidence_threshold=0.60,
            )
            return self._predictor
        except Exception as exc:  # Surface the genuine model failure to the UI.
            self._predictor_error = f"Could not load the configured LLM/RAG predictor: {exc}"
            raise RuntimeError(self._predictor_error) from exc

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "grid": {
                "rows": self.n_rows,
                "columns": self.n_cols,
                "classes": len(self.char_list),
                "keys": [str(key) for key in self.char_list],
            },
            "processed_data_available": REGISTRY_PATH.exists() and MODEL_PATH.exists(),
            "language_model": "loads on first run",
            "custom_text_notice": "Custom text uses simulated target-conditioned EEG; Study-D replay uses recorded epochs.",
        }

    def samples(self) -> list[dict[str, str]]:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        held_out = set(json.loads(TEST_SPLIT_PATH.read_text(encoding="utf-8")))
        samples = []
        for session_id in sorted(held_out):
            target = registry.get(session_id, "")
            fif = ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif"
            # The UI deliberately offers simple one-key-per-character targets.
            if fif.exists() and target and all(self._grid_key(char) is not None for char in target):
                samples.append({"id": session_id, "target": target})
        return samples[:18]

    def _grid_key(self, char: str) -> str | None:
        if char == " ":
            return "Sp"
        if char == ".":
            return "Prd"
        if char == "\n":
            return None
        candidate = char.upper()
        return candidate if candidate in self.char_list else None

    def _target_index(self, char: str) -> int:
        key = self._grid_key(char)
        if key is None:
            raise ValueError(f"{char!r} is not available on the Study-D keyboard")
        return self.char_list.index(key)

    def _prior(self, context: str) -> tuple[np.ndarray, dict[str, Any]]:
        predictor = self._load_predictor()
        prior = predictor.predict_next_char(context)
        diagnostics = asdict(predictor.last_diagnostics)
        top = np.argsort(prior)[-4:][::-1]
        return prior, {
            "top": [
                {"key": str(self.char_list[index]), "probability": round(float(prior[index]), 4)}
                for index in top
            ],
            "rag": {
                "enabled": diagnostics["rag_enabled"],
                "confidence": round(float(diagnostics["retrieval_confidence"]), 4),
                "global_weight": round(float(diagnostics["global_gate_weight"]), 4),
                "reason": diagnostics["reason"],
                "matches": diagnostics["matched_phrases"][:3],
            },
        }

    def _keys_for_stimulus(self, stimulus_code: int) -> list[str]:
        """Return the keyboard keys illuminated by one RC stimulus code."""
        if 1 <= stimulus_code <= self.n_rows:
            return [str(key) for key in self.matrix[stimulus_code - 1, :]]
        column = stimulus_code - self.n_rows - 1
        if 0 <= column < self.n_cols:
            return [str(key) for key in self.matrix[:, column]]
        return []

    def _flash_plan(self, flashes: list[dict[str, float]], prior: np.ndarray,
                    sequence: int) -> list[dict[str, Any]]:
        """Attach visible RC groups and LLM/RAG-aware ordering to flash scores.

        The first sequence is a complete row-first/column-second scan. Later
        sequences prioritize groups with the most next-character prior mass.
        Recorded EEG scores stay attached to their original stimulus code, so
        reordering the visual decision plan never changes decoder evidence.
        """
        plan = []
        for flash in flashes:
            code = int(flash["stimulus_code"])
            keys = self._keys_for_stimulus(code)
            key_indices = [self.char_list.index(key) for key in keys]
            group_mass = float(prior[key_indices].sum()) if key_indices else 0.0
            kind = "row" if code <= self.n_rows else "column"
            group_number = code if kind == "row" else code - self.n_rows
            plan.append({
                "stimulus_code": code,
                "kind": kind,
                "group_number": group_number,
                "keys": keys,
                "label": f"{kind.upper()} {group_number}",
                "priority_mass": round(group_mass, 5),
                "target_probability": round(float(flash["target_probability"]), 4),
                "classifier_target": bool(float(flash["target_probability"]) >= 0.5),
            })

        if sequence == 1:
            plan.sort(key=lambda flash: (flash["kind"] != "row", flash["group_number"]))
        else:
            plan.sort(key=lambda flash: (-flash["priority_mass"], flash["kind"] != "row", flash["group_number"]))
        return plan

    @staticmethod
    def _synthetic_eeg(target_index: int, class_count: int, seed: str, sequence: int) -> np.ndarray:
        """Deterministic simulated likelihood for text without recorded EEG.

        This is intentionally not presented as a recorded P300 measurement.
        The shape mirrors the decoder input: a full grid posterior per sequence.
        """
        digest = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        probabilities = rng.uniform(0.002, 0.014, class_count)
        probabilities[target_index] = 0.48 + min(sequence * 0.07, 0.34)
        probabilities /= probabilities.sum()
        return probabilities

    def _decode(self, message: str, evidence_provider, source: str, session_id: str | None = None):
        from src.models.decoder import P300Decoder
        from src.models.fusion import BayesianFusionEngine

        decoder = P300Decoder(self.matrix)
        fusion = BayesianFusionEngine(
            num_classes=len(self.char_list), mode="fixed", base_alpha=0.1
        )
        decoded = ""
        selections = []
        total_sequences = 0

        for char_index, target in enumerate(message):
            target_index = self._target_index(target)
            decoder.reset()
            prior, prior_payload = self._prior(decoded)
            decoder.accumulated_log_probs += fusion.get_initial_log_bias(prior)
            traces = []
            selected_key = None
            selected_confidence = 0.0

            for sequence in range(1, 16):
                eeg, raw_flashes = evidence_provider(char_index, target, target_index, sequence)
                decoder.accumulate_evidence(eeg)
                predicted_key, confidence = decoder.decode_character()
                target_evidence = float(eeg[target_index])
                traces.append({
                    "sequence": sequence,
                    "flash_mode": "standard_row_column" if sequence == 1 else "llm_rag_ranked",
                    "flashes": self._flash_plan(raw_flashes, prior, sequence),
                    "predicted_key": str(predicted_key),
                    "decoder_confidence": round(float(confidence), 4),
                    "target_evidence": round(target_evidence, 4),
                })
                selected_key, selected_confidence = predicted_key, confidence
                if sequence >= 2 and confidence >= 0.85:
                    break

            selected_text = " " if selected_key == "Sp" else ("." if selected_key == "Prd" else str(selected_key))
            decoded += selected_text
            total_sequences += len(traces)
            selections.append({
                "index": char_index,
                "target_key": self._grid_key(target),
                "target_character": target,
                "predicted_key": str(selected_key),
                "predicted_character": selected_text,
                "correct": selected_key == self._grid_key(target),
                "confidence": round(float(selected_confidence), 4),
                "sequences": len(traces),
                "trace": traces,
                "prior": prior_payload,
            })

        correct = sum(item["correct"] for item in selections)
        return {
            "source": source,
            "session_id": session_id,
            "target": message,
            "decoded": decoded,
            "selections": selections,
            "summary": {
                "characters": len(selections),
                "correct": correct,
                "accuracy": round(100 * correct / len(selections), 2) if selections else 0.0,
                "sequences": total_sequences,
                "mean_sequences_per_character": round(total_sequences / len(selections), 2) if selections else 0.0,
                "maximum_sequences_per_character": 15,
            },
            "evidence_notice": (
                "Recorded Study-D epochs scored by the saved SWLDA classifier."
                if source == "study_d_replay"
                else "Simulated target-conditioned EEG likelihoods; custom text has no recorded EEG."
            ),
        }

    def replay(self, session_id: str, scan_mode: str = "compact") -> dict[str, Any]:
        if scan_mode not in {"compact", "standard"}:
            raise ValueError("scan_mode must be 'compact' or 'standard'.")
        cache_key = (session_id, scan_mode)
        if cache_key in self._replay_cache:
            return self._replay_cache[cache_key]
        available = {sample["id"]: sample["target"] for sample in self.samples()}
        if session_id not in available:
            raise ValueError("Choose a session from the provided held-out Study-D samples.")

        from run_pipeline import real_eeg_classifier_flash_stream, real_eeg_classifier_stream

        eeg_path = ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif"
        classifier = self._load_classifier()

        if scan_mode == "compact":
            result = self._decode_row_focus(
                available[session_id], str(eeg_path), classifier, session_id
            )
        else:
            def evidence(char_index, _target, _target_index, sequence):
                posterior = real_eeg_classifier_stream(
                    str(eeg_path), char_index, sequence, classifier, self.char_list,
                    self.n_rows, self.n_cols, self.n_rows + self.n_cols,
                )
                flashes = real_eeg_classifier_flash_stream(
                    str(eeg_path), char_index, sequence, classifier,
                    self.n_rows, self.n_cols, self.n_rows + self.n_cols,
                )
                return posterior, flashes

            result = self._decode(
                available[session_id], evidence, "study_d_replay", session_id
            )
        self._replay_cache[cache_key] = result
        return result

    def _decode_row_focus(self, message: str, eeg_path: str, classifier,
                          session_id: str) -> dict[str, Any]:
        """Replay a compact two-pass scan over the active top five rows.

        Pass one presents rows 1--5 once.  Pass two presents the individual
        keys in the best row, plus the runner-up when its row posterior is
        within ``ROW_CLOSE_MARGIN``.  Rows 6--9 remain visible in the UI but
        deliberately receive no evidence or flashes.
        """
        from run_pipeline import real_eeg_classifier_flash_stream

        ACTIVE_ROWS = min(5, self.n_rows)
        ROW_CLOSE_MARGIN = 0.08
        decoded, selections, total_flashes = "", [], 0
        for char_index, target in enumerate(message):
            prior, prior_payload = self._prior(decoded)
            grid_prior = prior.reshape(self.n_rows, self.n_cols)
            flashes = real_eeg_classifier_flash_stream(
                eeg_path, char_index, 1, classifier, self.n_rows, self.n_cols,
                self.n_rows + self.n_cols,
            )
            scores = {int(flash["stimulus_code"]): float(flash["target_probability"])
                      for flash in flashes}
            active_row_prior = grid_prior[:ACTIVE_ROWS].sum(axis=1)
            row_log = np.log(np.clip(active_row_prior, 1e-9, 1.0))
            row_flashes = []
            row_likelihoods = {}
            for row in range(ACTIVE_ROWS):
                code = row + 1
                probability = np.clip(scores.get(code, 0.5), 1e-6, 1 - 1e-6)
                row_likelihoods[row] = probability
                row_log[row] += np.log(probability) - np.log1p(-probability)
                row_flashes.append({
                    "stimulus_code": code, "kind": "row", "group_number": code,
                    "keys": [str(key) for key in self.matrix[row, :]],
                    "label": f"ROW {code}", "priority_mass": round(float(active_row_prior[row]), 5),
                    "target_probability": round(float(probability), 4),
                    "classifier_target": bool(probability >= 0.5),
                })
            row_post = np.exp(row_log - row_log.max())
            row_post /= row_post.sum()
            ranked_rows = list(np.argsort(row_post)[::-1])
            selected_rows = [int(ranked_rows[0])]
            if len(ranked_rows) > 1 and row_post[ranked_rows[0]] - row_post[ranked_rows[1]] <= ROW_CLOSE_MARGIN:
                selected_rows.append(int(ranked_rows[1]))

            candidate_log = np.full((self.n_rows, self.n_cols), -np.inf)
            key_flashes = []
            for row in selected_rows:
                for col in range(self.n_cols):
                    code = self.n_rows + col + 1
                    probability = np.clip(scores.get(code, 0.5), 1e-6, 1 - 1e-6)
                    candidate_log[row, col] = (
                        np.log(max(grid_prior[row, col], 1e-9))
                        + np.log(row_likelihoods[row]) - np.log1p(-row_likelihoods[row])
                        + np.log(probability) - np.log1p(-probability)
                    )
                    key = str(self.matrix[row, col])
                    key_flashes.append({
                        "stimulus_code": code, "kind": "key", "group_number": col + 1,
                        "keys": [key], "label": f"ROW {row + 1} · {key}",
                        "priority_mass": round(float(grid_prior[row, col]), 5),
                        "target_probability": round(float(probability), 4),
                        "classifier_target": bool(probability >= 0.5),
                    })
            flat_log = candidate_log.ravel()
            posterior = np.exp(flat_log - np.max(flat_log))
            posterior /= posterior.sum()
            selected_index = int(np.argmax(posterior))
            selected_key = self.char_list[selected_index]
            selected_text = " " if selected_key == "Sp" else ("." if selected_key == "Prd" else str(selected_key))
            decoded += selected_text
            target_row, target_col = divmod(self._target_index(target), self.n_cols)
            confidence = float(posterior[selected_index])
            used = len(row_flashes) + len(key_flashes)
            total_flashes += used
            selections.append({
                "index": char_index, "target_key": self._grid_key(target),
                "target_character": target, "predicted_key": str(selected_key),
                "predicted_character": selected_text,
                "correct": (selected_index == self._target_index(target)),
                "confidence": round(confidence, 4), "sequences": 2,
                "flashes_used": used, "active_rows": [row + 1 for row in selected_rows],
                "trace": [
                    {"sequence": 1, "flash_mode": "rows_1_to_5_shortlist", "flashes": row_flashes,
                     "predicted_key": f"ROW {selected_rows[0] + 1}",
                     "decoder_confidence": round(float(row_post[selected_rows[0]]), 4),
                     "target_evidence": round(float(row_post[target_row]) if target_row < ACTIVE_ROWS else 0.0, 4)},
                    {"sequence": 2, "flash_mode": "shortlisted_row_keys", "flashes": key_flashes,
                     "predicted_key": str(selected_key), "decoder_confidence": round(confidence, 4),
                     "target_evidence": round(float(posterior[self._target_index(target)]), 4)},
                ], "prior": prior_payload,
            })
        correct = sum(item["correct"] for item in selections)
        count = len(selections)
        return {
            "source": "study_d_row_focus_replay", "session_id": session_id,
            "target": message, "decoded": decoded, "selections": selections,
            "summary": {"characters": count, "correct": correct,
                        "accuracy": round(100 * correct / count, 2) if count else 0.0,
                        "sequences": total_flashes,
                        "mean_flashes_per_character": round(total_flashes / count, 2) if count else 0.0,
                        "mean_sequences_per_character": round(total_flashes / count, 2) if count else 0.0,
                        "maximum_sequences_per_character": 21},
            "evidence_notice": "Compact replay: rows 1–5 flash once, then only key(s) in the shortlisted row(s) flash. Bottom rows remain visible but inactive.",
        }

    def flash_trace(self, session_id: str, char_index: int, sequence: int,
                    stimulus_code: int) -> dict[str, Any]:
        """Expose a recorded single-flash trace for the replay waveform."""
        if session_id not in {sample["id"] for sample in self.samples()}:
            raise ValueError("Unknown session_id.")
        from run_pipeline import real_eeg_single_flash_trace

        trace = real_eeg_single_flash_trace(
            str(ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif"),
            char_index, sequence, stimulus_code, self.n_rows, self.n_cols,
            self.n_rows + self.n_cols,
        )
        if trace is None:
            raise ValueError("No recorded epoch for that flash.")
        return trace

    def simulate(self, message: str) -> dict[str, Any]:
        message = message.strip().upper()
        if not message:
            raise ValueError("Enter a sentence or word to simulate.")
        if len(message) > 80:
            raise ValueError("Custom text is limited to 80 characters for the interactive demo.")
        unsupported = sorted({char for char in message if self._grid_key(char) is None})
        if unsupported:
            raise ValueError(f"Unsupported keyboard character(s): {', '.join(repr(char) for char in unsupported)}")

        def evidence(char_index, _target, target_index, sequence):
            posterior = self._synthetic_eeg(
                target_index, len(self.char_list), message + str(char_index), sequence
            )
            target_row, target_col = divmod(target_index, self.n_cols)
            target_codes = {target_row + 1, self.n_rows + target_col + 1}
            flashes = []
            for code in range(1, self.n_rows + self.n_cols + 1):
                # The simulation's individual flash scores remain consistent
                # with its target-conditioned grid posterior.
                score = 0.72 if code in target_codes else 0.03
                flashes.append({"stimulus_code": code, "target_probability": score})
            return posterior, flashes

        return self._decode(message, evidence, "custom_simulation")


SERVICE = ReplayService()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROTOTYPE_DIR), **kwargs)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/health":
                return self._json(SERVICE.health())
            if path == "/api/samples":
                return self._json({"samples": SERVICE.samples()})
            if path == "/api/flash_trace":
                query = parse_qs(urlparse(self.path).query)
                return self._json(SERVICE.flash_trace(
                    query.get("session_id", [""])[0], int(query.get("char_index", ["0"])[0]),
                    int(query.get("sequence", ["1"])[0]), int(query.get("stimulus_code", ["0"])[0]),
                ))
            if path == "/":
                self.path = "/index.html"
            return super().do_GET()
        except Exception as exc:
            return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/replay":
                return self._json(SERVICE.replay(
                    str(payload.get("session_id", "")),
                    str(payload.get("scan_mode", "compact")),
                ))
            if path == "/api/simulate":
                return self._json(SERVICE.simulate(str(payload.get("message", ""))))
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format, *args):
        print(f"[prototype] {format % args}")


if __name__ == "__main__":
    port = int(os.environ.get("P300_PROTOTYPE_PORT", "8000"))
    print(f"Neural Type prototype: http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
