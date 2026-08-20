"""
Day 5 — LLM + retrieval (RAG) predictive layer
Lightweight LM (distilGPT2 / T5-small) producing next-character priors,
conditioned on real previously-decoded context. Retrieval layer over a
public AAC/common-phrase corpus for simulated personalization.
Fill in during Day 5.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMPredictor:
    def __init__(self, spelling_matrix, model_name='distilgpt2', temperature=1.5):
        """
        Initializes the lightweight language model for next-character prediction.
        """
        self.grid_matrix = spelling_matrix
        self.char_list = list(self.grid_matrix.flatten())
        self.temperature = temperature
        
        # Load lightweight LM
        # Suppress the warning logs from transformers
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Create a mapping from characters to token IDs
        self.token_mapping = self._build_token_mapping()

    def _build_token_mapping(self):
        """Maps grid characters to GPT-2 token IDs. Multi-character special-key
        labels (PgUp, LfAw, email, etc.) don't correspond to a single meaningful
        token -- exclude them from LM scoring rather than silently using a
        garbage first-token match."""
        mapping = {}
        for char in self.char_list:
            mapped_char = ' ' if char == '_' else char
            if len(mapped_char) > 1 and mapped_char != ' ':
                continue  # skip multi-char special keys entirely
            tokens = self.tokenizer.encode(' ' + mapped_char.upper(), add_special_tokens=False)
            if tokens:
                mapping[char] = tokens[0]
        return mapping

    def predict_next_char(self, context_so_far):
        """
        Takes the spelled string context and returns a 36-element probability 
        distribution over the spelling matrix.
        """
        # If no context (start of word), return a flat, uniform distribution
        if not context_so_far:
            probs = np.ones(len(self.char_list))
            return probs / probs.sum()

        # Tokenize context
        clean_context = context_so_far.replace('_', ' ').lower()
        input_ids = self.tokenizer.encode(clean_context, return_tensors='pt')
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Get logits for the very last token in the sequence
            next_token_logits = outputs.logits[0, -1, :]
        
        # Apply temperature scaling (higher temperature = flatter, less overconfident distribution)
        next_token_logits = next_token_logits / self.temperature
        
        # Extract logits specifically for our 36 characters
        grid_logits = np.zeros(len(self.char_list))
        unmapped_mask = np.zeros(len(self.char_list), dtype=bool)

        for idx, char in enumerate(self.char_list):
            if char in self.token_mapping:
                token_id = self.token_mapping[char]
                grid_logits[idx] = next_token_logits[token_id].item()
            else:
                unmapped_mask[idx] = True

        grid_logits = grid_logits - np.max(grid_logits[~unmapped_mask]) if (~unmapped_mask).any() else grid_logits
        probs = np.exp(grid_logits)
        probs[unmapped_mask] = 0.0
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(self.char_list)) / len(self.char_list)

        return probs