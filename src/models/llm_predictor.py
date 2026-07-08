"""
Day 5 — LLM + retrieval (RAG) predictive layer
Lightweight LM (distilGPT2 / T5-small) producing next-character priors,
conditioned on real previously-decoded context. Retrieval layer over a
public AAC/common-phrase corpus for simulated personalization.
Fill in during Day 5.
"""
import numpy as np
import torch
try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
except ImportError:
    # Fallback/mock if transformers isn't installed locally yet
    GPT2LMHeadModel = None

class DistilGPT2Predictor:
    def __init__(self, grid_characters: list, temperature: float = 1.5, device: str = 'cpu'):
        """
        Generates next-character probabilities based on linguistic context.
        
        Parameters:
        - grid_characters: 1D list of characters supported by the current BCI grid.
        - temperature: Softmax temperature (higher = flatter distribution, less overconfident).
        - device: 'cpu', 'cuda', or 'mps' (Apple Silicon).
        """
        self.grid_characters = grid_characters
        self.num_classes = len(grid_characters)
        self.temperature = temperature
        self.device = device
        
        # Mapping from characters to their grid index
        self.char_to_idx = {char: idx for idx, char in enumerate(self.grid_characters)}
        
        if GPT2LMHeadModel is not None:
            self.tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
            self.model = GPT2LMHeadModel.from_pretrained("distilgpt2").to(self.device)
            self.model.eval()
            self._map_grid_to_tokens()
        else:
            print("WARNING: Transformers not installed. LLM Predictor will return uniform distribution.")
            self.model = None

    def _map_grid_to_tokens(self):
        """
        Maps our specific grid characters to the LLM's token IDs.
        GPT-2 tokenization is quirky (spaces matter). We map the raw character 
        as well as common prefixes (like space + char).
        """
        self.grid_token_ids = {}
        for char in self.grid_characters:
            # We look up the token ID for the character itself (lower and upper)
            # as well as with a leading space (standard GPT-2 BPE behavior)
            tokens = [
                self.tokenizer.encode(char, add_special_tokens=False)[0],
                self.tokenizer.encode(char.lower(), add_special_tokens=False)[0],
                self.tokenizer.encode(" " + char, add_special_tokens=False)[0],
                self.tokenizer.encode(" " + char.lower(), add_special_tokens=False)[0]
            ]
            # Store unique valid tokens for this character
            self.grid_token_ids[char] = list(set(tokens))

    def get_prior(self, context: str) -> np.ndarray:
        """
        Given the current spelled context, returns a normalized probability 
        distribution over the specific grid characters.
        """
        if not self.model or not context:
            # Return uniform if no context or no model
            return np.ones(self.num_classes) / self.num_classes
            
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the next token prediction
            next_token_logits = outputs.logits[0, -1, :]
            
        # Scale by temperature
        scaled_logits = next_token_logits / self.temperature
        
        # Convert all vocab logits to probabilities
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1).cpu().numpy()
        
        # Slice out only the probabilities for our grid characters
        grid_probs = np.zeros(self.num_classes)
        for char, token_ids in self.grid_token_ids.items():
            idx = self.char_to_idx[char]
            # Sum probabilities of all BPE token variants for this character
            grid_probs[idx] = sum([probs[tid] for tid in token_ids])
            
        # Re-normalize over just our grid
        sum_probs = np.sum(grid_probs)
        if sum_probs > 0:
            grid_probs = grid_probs / sum_probs
        else:
            # Fallback if no probability mass landed on valid chars (rare)
            grid_probs = np.ones(self.num_classes) / self.num_classes
            
        # Integrate TF-IDF RAG here later in the sprint
        # For now, return pure LLM prior
        return grid_probs