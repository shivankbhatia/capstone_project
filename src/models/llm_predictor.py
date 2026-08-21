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
        

    def predict_next_char(self, context_so_far, num_beams=8):
        if not context_so_far:
            probs = np.ones(len(self.char_list))
            return probs / probs.sum()

        clean_context = context_so_far.replace('_', ' ').lower()
        prompt_ids = self.tokenizer.encode(clean_context, return_tensors='pt')

        with torch.no_grad():
            output = self.model.generate(
                prompt_ids,
                max_new_tokens=3,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        beam_weights = torch.softmax(output.sequences_scores, dim=0).numpy()

        grid_probs = np.zeros(len(self.char_list))
        for seq, weight in zip(output.sequences, beam_weights):
            generated_ids = seq[prompt_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids).lower()
            stripped = generated_text.lstrip(' ')
            if not stripped:
                continue

            next_char = stripped[0]
            grid_label = 'Sp' if next_char == ' ' else next_char

            for idx, char in enumerate(self.char_list):
                if char.lower() == grid_label.lower():
                    grid_probs[idx] += weight
                    break

        if grid_probs.sum() > 0:
            probs = grid_probs / grid_probs.sum()
        else:
            probs = np.ones(len(self.char_list)) / len(self.char_list)

        return probs