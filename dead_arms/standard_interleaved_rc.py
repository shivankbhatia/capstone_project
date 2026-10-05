"""Frozen snapshot of ``ReplayService._decode`` before dynamic-stop tuning.

Taken from prototype/server.py with confidence threshold 0.85, a two-sequence
minimum, and a 15-sequence cap.
"""


class ReplayServiceSnapshot:
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
