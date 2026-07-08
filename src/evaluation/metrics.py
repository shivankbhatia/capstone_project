"""
Day 7/9 — Evaluation metrics
Accuracy, information transfer rate (ITR, bits/min), effective WPM,
paired significance testing (Wilcoxon signed-rank). Fill in during Day 7/9.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class SpellerMetrics:
    total_characters: int = 0
    correct_characters: int = 0
    total_flashes_used: int = 0
    total_time_seconds: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct_characters / max(1, self.total_characters)
        
    @property
    def average_flashes_per_char(self) -> float:
        return self.total_flashes_used / max(1, self.total_characters)
        
    @property
    def time_per_character(self) -> float:
        return self.total_time_seconds / max(1, self.total_characters)
        
    @property
    def wpm(self) -> float:
        """Words per minute (assuming standard 5 chars per word)"""
        chars_per_min = 60.0 / max(0.001, self.time_per_character)
        return chars_per_min / 5.0

class AblationTracker:
    def __init__(self):
        self.results = {}
        
    def record_run(self, condition_name: str, metrics: SpellerMetrics, itr: float):
        """Stores the result of a specific pipeline condition."""
        self.results[condition_name] = {
            "Accuracy (%)": round(metrics.accuracy * 100, 2),
            "Flashes/Char": round(metrics.average_flashes_per_char, 2),
            "ITR (bits/min)": round(itr, 2),
            "WPM": round(metrics.wpm, 2)
        }
        
    def print_summary(self):
        print("\n" + "="*50)
        print("ABLATION STUDY RESULTS SUMMARY")
        print("="*50)
        for condition, data in self.results.items():
            print(f"\nCondition: [{condition}]")
            for metric, value in data.items():
                print(f"  {metric}: {value}")
        print("="*50)