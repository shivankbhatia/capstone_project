"""
Day 5 — LLM predictive layer
Lightweight LM (distilGPT2) producing next-character priors, via vocabulary
prefix-filtering: get the next-token distribution ONCE (conditioned on text
before the in-progress word), then for each candidate next character, sum
probability mass over every vocab token whose literal decoded text starts
with (partial_word + candidate) -- this avoids asking the model to judge
whether the in-progress word is "finished" (which it can't know), and
avoids per-candidate forward passes entirely.
"""
import bisect
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMPredictor:
    def __init__(self, spelling_matrix, model_name='distilgpt2'):
        self.grid_matrix = spelling_matrix
        self.char_list = list(self.grid_matrix.flatten())

        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

        self.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

        # One-time: sorted (decoded_text, token_id) index over the full
        # vocab, for fast prefix lookup via binary search.
        vocab_size = self.model.config.vocab_size
        entries = []
        for tid in range(vocab_size):
            stripped = self.tokenizer.decode([tid]).lstrip(' ').lower()
            if stripped:
                entries.append((stripped, tid))
        entries.sort()
        self._sorted_texts = [t for t, _ in entries]
        self._sorted_ids = [i for _, i in entries]

    def _matching_token_ids(self, prefix):
        lo = bisect.bisect_left(self._sorted_texts, prefix)
        hi = bisect.bisect_left(self._sorted_texts, prefix + '\uffff')
        return self._sorted_ids[lo:hi]

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

        input_ids = (self.tokenizer.encode(prior_text, return_tensors='pt')
                     if prior_text else torch.tensor([[self.bos_token_id]]))

        with torch.no_grad():
            logits = self.model(input_ids).logits[0, -1, :]
            token_probs = torch.softmax(logits, dim=-1).numpy()

        grid_probs = np.zeros(len(self.char_list))
        for idx, char in enumerate(self.char_list):
            mapped_char = ' ' if char == 'Sp' else char
            if len(mapped_char) != 1:
                continue
            candidate_prefix = (partial_word + mapped_char).lower()
            token_ids = self._matching_token_ids(candidate_prefix)
            if token_ids:
                grid_probs[idx] = token_probs[token_ids].sum()

        if grid_probs.sum() > 0:
            probs = grid_probs / grid_probs.sum()
        else:
            probs = np.ones(len(self.char_list)) / len(self.char_list)

        return probs