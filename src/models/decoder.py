"""
Day 4 — Character-level decoding harness
Row/column flash accumulation -> per-character posterior distribution.
No-LM baseline accuracy/ITR computed here. Fill in during Day 4.
"""

import numpy as np

class P300Decoder:
    def __init__(self, spelling_matrix):
        """
        Initializes the character decoder based on the grid matrix.
        """
        self.grid_matrix = np.array(spelling_matrix)
        self.num_classes = self.grid_matrix.size
        self.char_list = list(self.grid_matrix.flatten())
        
        # Internal state to hold probabilities across flash sequences
        self.accumulated_log_probs = np.zeros(self.num_classes)

    def reset(self):
        """
        CRITICAL: Resets the accumulated probabilities. 
        Must be called before starting a new character!
        """
        self.accumulated_log_probs = np.zeros(self.num_classes)

    def accumulate_evidence(self, probabilities):
        """
        Accumulates evidence from the latest sequence.
        Works in the log domain to prevent floating point underflow.
        """
        eps = 1e-9 # Prevent log(0)
        self.accumulated_log_probs += np.log(np.clip(probabilities, eps, 1.0))

    def decode_character(self):
        """
        Calculates the most likely character based on accumulated evidence.
        Returns: (predicted_char, confidence_score)
        """
        # Convert log probabilities back to normalized probabilities
        # using the logsumexp trick for numerical stability
        max_log = np.max(self.accumulated_log_probs)
        exp_probs = np.exp(self.accumulated_log_probs - max_log)
        normalized_probs = exp_probs / np.sum(exp_probs)

        best_idx = np.argmax(normalized_probs)
        best_char = self.char_list[best_idx]
        confidence = normalized_probs[best_idx]

        return best_char, confidence

def calculate_itr(num_classes, accuracy_pct, time_per_selection):
    """
    Calculates the Information Transfer Rate (bits/min) according to the 
    standard BCI formula (Wolpaw et al., 2002).
    """
    if time_per_selection <= 0:
        raise ValueError("Time per selection must be > 0")

    P = accuracy_pct / 100.0
    N = num_classes

    if P == 1.0:
        bits_per_selection = np.log2(N)
    elif P <= 0.0:
        bits_per_selection = 0.0
    else:
        bits_per_selection = np.log2(N) + P * np.log2(P) + (1 - P) * np.log2((1 - P) / (N - 1))

    # Ensure bits per selection doesn't go negative due to noise
    bits_per_selection = max(0.0, bits_per_selection)

    itr = bits_per_selection * (60.0 / time_per_selection)
    return itr