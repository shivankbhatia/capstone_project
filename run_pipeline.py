import numpy as np
import time
from src.models.decoder import P300Decoder, calculate_itr
from src.models.llm_predictor import DistilGPT2Predictor
from src.models.fusion import BayesianFusionEngine
from src.preprocessing.build_session_sequence import yield_character_trials
from src.evaluation.metrics import SpellerMetrics, AblationTracker

# 6x6 Standard Grid for bigP3BCI Study D
STUDY_D_GRID = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
]

def mock_eeg_classifier_stream(target_char, char_list):
    """
    Simulates the output of a calibrated SWLDA classifier.
    Safely handles characters that might not exist in the 36-class grid.
    """
    import numpy as np
    probs = np.random.uniform(0.01, 0.05, len(char_list))
    
    char_upper = target_char.upper()
    
    if char_upper in char_list:
        target_idx = char_list.index(char_upper)
        probs[target_idx] = 0.85  # Simulate high classifier confidence
    else:
        # If a character (like '0' or space) isn't in the default grid, 
        # we skip boosting it so the script doesn't crash.
        pass 
        
    probs /= probs.sum()
    return probs

def run_evaluation(
    decoder: P300Decoder, 
    llm: DistilGPT2Predictor, 
    fusion_engine: BayesianFusionEngine, 
    max_flashes: int = 15, 
    confidence_threshold: float = 0.85
) -> tuple:
    """Runs the end-to-end replay evaluation over the dataset."""
    metrics = SpellerMetrics()
    char_list = decoder.char_list
    
    # 1. Initialize trial generator
    metadata_path = "data/processed/ground_truth_registry.json"
    dataset_dir = "data/raw/bigP3BCI_dataset/"
    trials = yield_character_trials(metadata_path, dataset_dir)
    
    for trial in trials:
        target = trial['target_char']
        context = trial['context_so_far']
        
        start_time = time.time()
        
        # 2. Get Linguistic Prior (Run once per character)
        llm_prior = llm.get_prior(context)
        
        # 3. Flash Sequence Accumulation Loop
        accumulated_evidence = np.ones(decoder.num_classes) / decoder.num_classes
        flashes_used = 0
        predicted_char = None
        
        for flash_idx in range(1, max_flashes + 1):
            flashes_used += 1
            
            # Simulated EEG inference (replace with real epoching & classifier.predict_proba)
            eeg_posteriors = mock_eeg_classifier_stream(target, char_list)
            
            # Accumulate EEG evidence in log domain
            accumulated_evidence = np.log(accumulated_evidence) + np.log(eeg_posteriors)
            accumulated_evidence = np.exp(accumulated_evidence - np.max(accumulated_evidence))
            accumulated_evidence = accumulated_evidence / np.sum(accumulated_evidence)
            
            # 4. Bayesian Fusion
            fused_probs = fusion_engine.fuse(accumulated_evidence, llm_prior)
            
            # 5. Adaptive Stopping Check
            best_char, confidence = decoder.decode_character(fused_probs)
            if confidence >= confidence_threshold:
                predicted_char = best_char
                break
                
        # Fallback if threshold never reached
        if not predicted_char:
            predicted_char, _ = decoder.decode_character(fused_probs)
            
        # 6. Record Metrics
        elapsed_time = time.time() - start_time
        metrics.total_characters += 1
        metrics.total_time_seconds += elapsed_time
        metrics.total_flashes_used += flashes_used
        if predicted_char == target:
            metrics.correct_characters += 1
            
    # Calculate Final ITR
    # Note: Using standard 2.0 seconds per flash sequence as time metric
    avg_time_per_char = (metrics.total_flashes_used / max(1, metrics.total_characters)) * 2.0 
    if avg_time_per_char <= 0:
        print("⚠️ Failed to calculate time per char. Returning 0 ITR.")
        return metrics, 0.0
    itr = calculate_itr(decoder.num_classes, metrics.accuracy, avg_time_per_char)
    
    return metrics, itr

if __name__ == "__main__":
    print("Initializing components for Phase 6 Evaluation...")
    decoder = P300Decoder(STUDY_D_GRID)
    
    # Init LLM (MPS/CUDA if available, else CPU)
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    llm = DistilGPT2Predictor(decoder.char_list, device=device)
    
    tracker = AblationTracker()
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: Standard Baseline (No Fusion)
    # ---------------------------------------------------------
    print("\nRunning Baseline (EEG Only)...")
    baseline_fusion = BayesianFusionEngine(mode='fixed', base_alpha=0.0)
    baseline_fusion.toggle(False)
    
    base_metrics, base_itr = run_evaluation(decoder, llm, baseline_fusion, confidence_threshold=0.95)
    tracker.record_run("Baseline (No LLM)", base_metrics, base_itr)
    
    # ---------------------------------------------------------
    # EXPERIMENT 2: Fixed Weight Fusion (alpha=1.0)
    # ---------------------------------------------------------
    print("\nRunning Fixed Weight Fusion...")
    fixed_fusion = BayesianFusionEngine(mode='fixed', base_alpha=1.0)
    
    fixed_metrics, fixed_itr = run_evaluation(decoder, llm, fixed_fusion, confidence_threshold=0.85)
    tracker.record_run("Fusion (Fixed a=1.0)", fixed_metrics, fixed_itr)
    
    # ---------------------------------------------------------
    # EXPERIMENT 3: Adaptive Entropy Fusion
    # ---------------------------------------------------------
    print("\nRunning Adaptive Entropy Fusion...")
    adaptive_fusion = BayesianFusionEngine(mode='adaptive', base_alpha=1.5)
    
    adapt_metrics, adapt_itr = run_evaluation(decoder, llm, adaptive_fusion, confidence_threshold=0.85)
    tracker.record_run("Fusion (Adaptive)", adapt_metrics, adapt_itr)
    
    tracker.print_summary()