"""Frozen snapshot of ``ReplayService._decode_row_focus`` before tuning."""

import numpy as np


class ReplayServiceSnapshot:
    def _decode_row_focus(self, message: str, eeg_path: str, classifier,
                          session_id: str) -> dict[str, object]:
        """Replay a compact two-pass scan over the active top five rows.

        Pass one presents rows 1--5 once. Pass two presents the individual
        keys in the best row, plus the runner-up when its row posterior is
        within ``ROW_CLOSE_MARGIN``. Rows 6--9 remain visible in the UI but
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
