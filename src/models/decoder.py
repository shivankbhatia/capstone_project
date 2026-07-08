"""
Day 4 — Character-level decoding harness
Row/column flash accumulation -> per-character posterior distribution.
No-LM baseline accuracy/ITR computed here. Fill in during Day 4.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any

class P300Decoder:
    def __init__(self, grid_layout: List[List[str]]):
        """
        Initializes the decoder with a dynamic spelling matrix.
        
        Parameters:
        - grid_layout: 2D list of characters (e.g., 6x6 for Study D, or extended for Study E)
        """
        self.grid = grid_layout
        self.rows = len(grid_layout)
        self.cols = len(grid_layout[0])
        self.num_classes = self.rows * self.cols
        
        # Flatten grid for 1D probability array mapping (size 36 for 6x6)
        self.char_list = [char for row in self.grid for char in row]
        self.char_to_index = {char: idx for idx, char in enumerate(self.char_list)}

    def intersect_row_col_probs(self, row_probs: np.ndarray, col_probs: np.ndarray) -> np.ndarray:
        """
        Converts marginal row and column probabilities into a joint 1D character probability distribution.
        P(char_{i,j}) \propto P(row_i) * P(col_j)
        
        Parameters:
        - row_probs: 1D array of row probabilities (size R)
        - col_probs: 1D array of column probabilities (size C)
        
        Returns:
        - char_probs: 1D array of character probabilities (size R*C), normalized.
        """
        if len(row_probs) != self.rows or len(col_probs) != self.cols:
            raise ValueError(f"Expected {self.rows} rows and {self.cols} cols, got {len(row_probs)} and {len(col_probs)}")
            
        # Outer product yields the joint probability matrix assuming row/col independence
        joint_matrix = np.outer(row_probs, col_probs)
        
        # Flatten to match the 1D character list (size 36)
        flat_probs = joint_matrix.flatten()
        
        # Normalize to ensure it sums to 1.0 (handling minor floating point errors)
        return flat_probs / np.sum(flat_probs)

    def decode_character(self, accumulated_probs: np.ndarray) -> Tuple[str, float]:
        """
        Returns the character with the highest probability.
        
        Parameters:
        - accumulated_probs: 1D array of probabilities (size R*C)
        
        Returns:
        - (predicted_char, confidence_score)
        """
        best_idx = np.argmax(accumulated_probs)
        best_char = self.char_list[best_idx]
        confidence = accumulated_probs[best_idx]
        
        return best_char, float(confidence)

def calculate_itr(num_classes: int, accuracy: float, time_per_selection_sec: float) -> float:
    """
    Calculates the Information Transfer Rate (ITR) in bits per minute.
    
    Formula: B = log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))
    ITR = B / (time_per_selection_sec / 60)
    
    Parameters:
    - num_classes (N): Total number of targets (e.g., 36)
    - accuracy (P): Classification accuracy [0.0, 1.0]
    - time_per_selection_sec (T): Average time taken to select one character in seconds
    """
    if accuracy < 0.0 or accuracy > 1.0:
        raise ValueError("Accuracy must be between 0.0 and 1.0")
    if time_per_selection_sec <= 0:
        raise ValueError("Time per selection must be > 0")
        
    # Handle edge cases to prevent log(0)
    if accuracy == 1.0:
        bits_per_selection = math.log2(num_classes)
    elif accuracy == 0.0:
        bits_per_selection = math.log2(num_classes) + math.log2(1.0 / (num_classes - 1))
    else:
        term1 = math.log2(num_classes)
        term2 = accuracy * math.log2(accuracy)
        term3 = (1.0 - accuracy) * math.log2((1.0 - accuracy) / (num_classes - 1))
        bits_per_selection = term1 + term2 + term3
        
    # If ITR becomes negative (accuracy worse than random chance), clamp to 0
    bits_per_selection = max(0.0, bits_per_selection)
    
    # Convert bits/selection to bits/minute
    selections_per_minute = 60.0 / time_per_selection_sec
    return bits_per_selection * selections_per_minute