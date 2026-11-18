#!/usr/bin/env python3
"""Local API and static server for the Neural Type prototype.

The server has two deliberately distinct execution paths:

* ``study_d_replay`` replays a processed Study-D session through the saved
  SWLDA classifier, an optional fixed language-model prior, Bayesian fusion engine, and P300 decoder.
* ``custom_simulation`` uses the same grid, optional language prior, fusion engine, and
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
import errno
import os
import sys
import threading
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
FALLBACK_GRID_PATH = ROOT / "configs/studyd_grid_layout.json"
SPECIAL_KEY_MAP_PATH = ROOT / "configs/llm/special_key_map.json"
LOCKED_SPECIAL_KEY_MAP_SHA256 = "eb1eadd6e21ab4c290611a97284a4745e9bd329ec1a9c85c2f3f70d19397088e"
REGISTRY_PATH = ROOT / "data/processed/ground_truth_registry.json"
TEST_SPLIT_PATH = ROOT / "data/processed/test_sessions.json"
GROUND_TRUTH_VAULT_PATH = ROOT / "data/evaluation/ground_truth_vault.json"
MODEL_PATH = ROOT / "data/processed/clean_m0_epoch_scorer.pkl"
FIXED_LM_ALPHA = 0.85
SUPPORTED_ARMS = {"eeg_only", "fixed_lm"}
MAX_REQUEST_BYTES = 16 * 1024

class ReplayService:
    """Thin adapter around the repository's evaluation components."""

    def __init__(self, allow_heldout_eval: bool = False) -> None:
        self.allow_heldout_eval = allow_heldout_eval
        self.matrix, self.n_rows, self.n_cols, self.using_fallback_grid = self._load_matrix()
        self.char_list = list(self.matrix.flatten())
        self._classifier = None
        self._predictor = None
        self._predictor_error = None
        self._replay_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._fixed_block_indexing_cache: dict[str, bool] = {}
        self._load_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    @staticmethod
    def _load_matrix():
        from run_pipeline import load_spelling_matrix

        if not GRID_PATH.exists():
            matrix, rows, columns = load_spelling_matrix(str(FALLBACK_GRID_PATH), study_name="StudyD")
            return matrix, rows, columns, True
        matrix, rows, columns = load_spelling_matrix(str(GRID_PATH), study_name="StudyD")
        return matrix, rows, columns, False

    def _load_classifier(self):
        with self._load_lock:
            if self._classifier is None:
                self._classifier = joblib.load(MODEL_PATH)
        return self._classifier

    def _uses_fixed_block_indexing(self, eeg_path: Path) -> bool:
        """Return true when a session lacks adaptive sequence metadata."""
        path = str(eeg_path)
        if path not in self._fixed_block_indexing_cache:
            import mne

            metadata = mne.read_epochs(path, preload=False, verbose=False).metadata
            self._fixed_block_indexing_cache[path] = (
                metadata is None
                or not {"char_index", "sequence_in_char"}.issubset(metadata.columns)
            )
        return self._fixed_block_indexing_cache[path]

    def _load_predictor(self):
        """Load the fixed-LM prior once. RAG is intentionally unavailable here."""
        with self._load_lock:
            if self._predictor is not None:
                return self._predictor
            if self._predictor_error is not None:
                raise RuntimeError(self._predictor_error)
            try:
                from src.models.llm_predictor import LLMPredictor
                self._predictor = LLMPredictor(self.matrix, local_files_only=True)
                return self._predictor
            except Exception as exc:  # Surface the genuine model failure to the UI.
                self._predictor_error = f"Could not load the fixed language-model prior: {exc}"
                raise RuntimeError(self._predictor_error) from exc

    def health(self) -> dict[str, Any]:
        fixed_lm_available = self._fixed_lm_available()
        special_key_map_hash = hashlib.sha256(SPECIAL_KEY_MAP_PATH.read_bytes()).hexdigest()
        return {
            "status": "ready",
            "grid": {
                "rows": self.n_rows,
                "columns": self.n_cols,
                "classes": len(self.char_list),
                "keys": [str(key) for key in self.char_list],
            },
            "processed_data_available": REGISTRY_PATH.exists() and MODEL_PATH.exists(),
            "using_fallback_grid": self.using_fallback_grid,
            "heldout_evaluation_enabled": self.allow_heldout_eval,
            "language_model": "available offline" if fixed_lm_available else "not available offline",
            "fixed_lm_available_offline": fixed_lm_available,
            "supported_arms": ["eeg_only", "fixed_lm"],
            "fixed_lm": {"alpha": FIXED_LM_ALPHA, "provenance": "clean-M0 ablation ladder lock"},
            "locked_special_key_map": {
                "sha256": special_key_map_hash,
                "matches_lock": special_key_map_hash == LOCKED_SPECIAL_KEY_MAP_SHA256,
            },
            "custom_text_notice": "Custom text uses simulated target-conditioned EEG; Study-D replay uses recorded epochs.",
        }

    @staticmethod
    def _fixed_lm_available() -> bool:
        """Preflight the fixed LM without downloading model weights."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            AutoTokenizer.from_pretrained("distilgpt2", local_files_only=True)
            AutoModelForCausalLM.from_pretrained("distilgpt2", local_files_only=True)
            return True
        except Exception:
            return False

    def samples(self) -> list[dict[str, str]]:
        if not REGISTRY_PATH.exists() or not TEST_SPLIT_PATH.exists():
            return []
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        held_out = set(json.loads(TEST_SPLIT_PATH.read_text(encoding="utf-8")))
        if self.allow_heldout_eval:
            if not GROUND_TRUTH_VAULT_PATH.exists():
                raise RuntimeError("Held-out evaluation was enabled, but the label vault is unavailable.")
            targets = json.loads(GROUND_TRUTH_VAULT_PATH.read_text(encoding="utf-8"))
        else:
            targets = registry
        samples = []
        for session_id in sorted(held_out):
            target = targets.get(session_id, "")
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
        top = np.argsort(prior)[-4:][::-1]
        return prior, {
            "top": [
                {"key": str(self.char_list[index]), "probability": round(float(prior[index]), 4)}
                for index in top
            ],
            "arm": "fixed_lm",
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
        """Attach visible groups without changing the recorded flash order."""
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

    def _decode(self, message: str, evidence_provider, source: str, arm: str,
                session_id: str | None = None, confidence_threshold: float = 0.80,
                min_sequences: int = 2, max_sequences: int = 10):
        from src.models.decoder import P300Decoder
        from src.models.fusion import BayesianFusionEngine

        decoder = P300Decoder(self.matrix)
        fusion = BayesianFusionEngine(
            num_classes=len(self.char_list), mode="fixed",
            base_alpha=FIXED_LM_ALPHA if arm == "fixed_lm" else 0.0,
        )
        decoded = ""
        selections = []
        total_sequences = 0
        total_flashes = 0

        for char_index, target in enumerate(message):
            target_index = self._target_index(target)
            decoder.reset()
            if arm == "fixed_lm":
                prior, prior_payload = self._prior(decoded)
                decoder.accumulated_log_probs += fusion.get_initial_log_bias(prior)
            else:
                prior = np.full(len(self.char_list), 1 / len(self.char_list))
                prior_payload = {"top": [], "arm": "eeg_only"}
            traces = []
            selected_key = None
            selected_confidence = 0.0

            for sequence in range(1, max_sequences + 1):
                eeg, raw_flashes = evidence_provider(char_index, target, target_index, sequence)
                decoder.accumulate_evidence(eeg)
                predicted_key, confidence = decoder.decode_character()
                target_evidence = float(eeg[target_index])
                traces.append({
                    "sequence": sequence,
                    "flash_mode": "recorded_order" if source == "study_d_replay" else "simulated_order",
                    "flashes": self._flash_plan(raw_flashes, prior, sequence),
                    "predicted_key": str(predicted_key),
                    "decoder_confidence": round(float(confidence), 4),
                    "target_evidence": round(target_evidence, 4),
                })
                selected_key, selected_confidence = predicted_key, confidence
                if sequence >= min_sequences and confidence >= confidence_threshold:
                    break

            selected_text = " " if selected_key == "Sp" else ("." if selected_key == "Prd" else str(selected_key))
            decoded += selected_text
            total_sequences += len(traces)
            flashes_used = sum(len(trace["flashes"]) for trace in traces)
            total_flashes += flashes_used
            selections.append({
                "index": char_index,
                "target_key": self._grid_key(target),
                "target_character": target,
                "predicted_key": str(selected_key),
                "predicted_character": selected_text,
                "correct": selected_key == self._grid_key(target),
                "confidence": round(float(selected_confidence), 4),
                "sequences": len(traces),
                "flashes_used": flashes_used,
                "trace": traces,
                "prior": prior_payload,
            })

        correct = sum(item["correct"] for item in selections)
        return {
            "source": source,
            "arm": arm,
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
                "flashes": total_flashes,
                "mean_flashes_per_character": round(total_flashes / len(selections), 2) if selections else 0.0,
                "maximum_sequences_per_character": max_sequences,
            },
            "evidence_notice": (
                "Recorded Study-D epochs scored by the saved SWLDA classifier."
                if source == "study_d_replay"
                else "Simulated target-conditioned EEG likelihoods; custom text has no recorded EEG."
            ),
        }

    def replay(self, session_id: str, arm: str = "fixed_lm",
               confidence_threshold: float = 0.80, min_sequences: int = 2,
               max_sequences: int = 10) -> dict[str, Any]:
        if arm not in SUPPORTED_ARMS:
            raise ValueError("arm must be 'eeg_only' or 'fixed_lm'.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if min_sequences < 1 or max_sequences < min_sequences:
            raise ValueError("Require 1 <= min_sequences <= max_sequences.")
        cache_key = (session_id, arm, confidence_threshold, min_sequences, max_sequences)
        with self._cache_lock:
            if cache_key in self._replay_cache:
                return self._replay_cache[cache_key]
        available = {sample["id"]: sample["target"] for sample in self.samples()}
        if session_id not in available:
            raise ValueError("Choose a session from the provided held-out Study-D samples.")

        from run_pipeline import real_eeg_classifier_flash_stream, real_eeg_classifier_stream

        eeg_path = ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif"
        classifier = self._load_classifier()

        effective_max_sequences = (
            min(max_sequences, 15)
            if self._uses_fixed_block_indexing(eeg_path)
            else max_sequences
        )

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
            available[session_id], evidence, "study_d_replay", arm, session_id,
            confidence_threshold=confidence_threshold,
            min_sequences=min_sequences,
            max_sequences=effective_max_sequences,
        )
        with self._cache_lock:
            self._replay_cache[cache_key] = result
        return result

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

    def simulate(self, message: str, arm: str = "fixed_lm") -> dict[str, Any]:
        if arm not in SUPPORTED_ARMS:
            raise ValueError("arm must be 'eeg_only' or 'fixed_lm'.")
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

        return self._decode(message, evidence, "custom_simulation", arm)


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

    def _error(self, status: HTTPStatus, message: str):
        return self._json({"error": message}, status)

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
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Requested data or model is unavailable.")
        except Exception:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected server error.")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json.")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise ValueError("Content-Length must be an integer.") from exc
            if length < 0:
                raise ValueError("Content-Length must not be negative.")
            if length > MAX_REQUEST_BYTES:
                return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large.")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object.")
            if "scan_mode" in payload:
                raise ValueError("scan_mode was removed; standard full-grid replay is the only shipped mode.")
            if path == "/api/replay":
                return self._json(SERVICE.replay(
                    str(payload.get("session_id", "")),
                    str(payload.get("arm", "fixed_lm")),
                ))
            if path == "/api/simulate":
                return self._json(SERVICE.simulate(
                    str(payload.get("message", "")), str(payload.get("arm", "fixed_lm"))
                ))
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "Requested data or model is unavailable.")
        except Exception:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Unexpected server error.")

    def log_message(self, format, *args):
        print(f"[prototype] {format % args}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serve the Neural Type prototype.")
    parser.add_argument(
        "--allow-heldout-eval", action="store_true",
        help="Explicitly allow the GUI to read frozen held-out targets for a replay.",
    )
    args = parser.parse_args()
    SERVICE = ReplayService(allow_heldout_eval=args.allow_heldout_eval)
    port = int(os.environ.get("P300_PROTOTYPE_PORT", "8000"))
    print(f"Neural Type prototype: http://127.0.0.1:{port}")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Port {port} is already in use. Set P300_PROTOTYPE_PORT to another local port."
            ) from exc
        raise
