"""
Day 5 — LLM predictive layer
Lightweight LM (distilGPT2) produces next-key priors by scoring the full
candidate continuation under the model. Multi-character grid keys use an
explicit semantic text map, and all candidate token probabilities are summed
in log space. Candidate scoring is batched to avoid a forward pass per key.
"""
import json
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMPredictor:
    def __init__(self, spelling_matrix, model_name='distilgpt2', *,
                 local_files_only=False, special_key_map_path=None):
        self.grid_matrix = spelling_matrix
        self.char_list = list(self.grid_matrix.flatten())
        map_path = Path(special_key_map_path or Path(__file__).resolve().parents[2]
                        / "configs/llm/special_key_map.json")
        self.special_key_map = json.loads(map_path.read_text(encoding="utf-8"))

        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model.eval()

        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    def _candidate_text(self, key):
        """Render one grid key as text understood by the language model."""
        if key in self.special_key_map:
            return self.special_key_map[key]
        return str(key).lower()

    def _continuation_log_probabilities(self, prior_text, completions):
        """Batch-score every token in complete candidate continuations.

        Unlike the former first-token prefix lookup, this sums conditional
        log-probability across the full mapped key string, including keys that
        tokenize into several words or subword pieces.
        """
        prefix_ids = self.tokenizer.encode(prior_text, add_special_tokens=False) if prior_text else []
        if not prefix_ids:
            prefix_ids = [self.bos_token_id]
        encoded = []
        for completion in completions:
            full_ids = (
                self.tokenizer.encode(prior_text + completion, add_special_tokens=False)
                if prior_text else []
            )
            # A whitespace-terminated prior is a token boundary in GPT-2 BPE.
            # For tokenizers that merge across it, append the candidate tokens.
            if not full_ids or full_ids[:len(prefix_ids)] != prefix_ids:
                full_ids = prefix_ids + self.tokenizer.encode(
                    completion, add_special_tokens=False
                )
            encoded.append(full_ids)
        lengths = [len(ids) for ids in encoded]
        if not lengths:
            return np.empty(0, dtype=np.float64)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None) or 0
        max_length = max(lengths)
        input_array = np.full((len(encoded), max_length), pad_id, dtype=np.int64)
        attention_array = np.zeros((len(encoded), max_length), dtype=np.int64)
        for i, ids in enumerate(encoded):
            input_array[i, :len(ids)] = ids
            attention_array[i, :len(ids)] = 1
        input_ids = torch.as_tensor(input_array, dtype=torch.long)
        attention_mask = torch.as_tensor(attention_array, dtype=torch.long)
        with torch.no_grad():
            token_log_probs = torch.log_softmax(
                self.model(input_ids, attention_mask=attention_mask).logits, dim=-1
            )
        scores = []
        target_start = len(prefix_ids)
        for i, ids in enumerate(encoded):
            if len(ids) <= target_start:
                scores.append(float("-inf"))
                continue
            positions = torch.arange(target_start - 1, len(ids) - 1, dtype=torch.long)
            targets = torch.tensor(ids[target_start:], dtype=torch.long)
            scores.append(float(token_log_probs[i, positions, targets].sum().item()))
        return np.asarray(scores, dtype=np.float64)

    def _continuation_log_probability(self, prior_text, completion):
        """Score every token in a complete candidate continuation."""
        return float(self._continuation_log_probabilities(prior_text, [completion])[0])

    def predict_next_char(self, context_so_far):
        if not context_so_far:
            probs = np.ones(len(self.char_list))
            return probs / probs.sum()

        clean_context = context_so_far.replace('_', ' ').lower()
        last_space = clean_context.rfind(' ')
        if last_space == -1:
            prior_text, partial_word = '', clean_context
        else:
            prior_text, partial_word = clean_context[:last_space + 1], clean_context[last_space + 1:]

        completions = [partial_word + self._candidate_text(char) for char in self.char_list]
        candidate_log_probs = self._continuation_log_probabilities(prior_text, completions)

        finite = np.isfinite(candidate_log_probs)
        if finite.any():
            shifted = candidate_log_probs[finite] - np.max(candidate_log_probs[finite])
            grid_probs = np.zeros(len(self.char_list), dtype=np.float64)
            grid_probs[finite] = np.exp(shifted)
            probs = grid_probs / grid_probs.sum()
        else:
            probs = np.ones(len(self.char_list)) / len(self.char_list)

        return probs
