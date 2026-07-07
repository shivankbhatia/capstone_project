import json
import numpy as np
import mne
from pathlib import Path
from joblib import load
from src.models.decoder import P300Decoder

def evaluate_baseline(data_dir, registry_path, model_path):
    """
    Runs the full baseline evaluation on the processed dataset.
    """
    print("Loading Ground Truth Registry...")
    with open(registry_path, 'r') as f:
        registry = json.load(f)
        
    print("Loading Calibrated Classifier...")
    classifier = load(model_path)
    
    total_characters = 0
    correct_predictions = 0
    
    # BCI metrics
    flashes_per_sequence = 12 # Adjust if Study E uses a different number
    sequences_per_char = 10   # Example: 10 sequences per character selection
    flash_duration_ms = 100   # Typical flash + SOA duration
    seconds_per_char = (flashes_per_sequence * sequences_per_char * flash_duration_ms) / 1000.0

    for file_stem, metadata in registry.items():
        print(f"\nEvaluating Session: {file_stem}")
        
        # Determine grid size dynamically based on Study type
        study_type = metadata.get('study', 'StudyD')
        if study_type == 'StudyE':
            # Example extended matrix for Study E (Update with exact dimensions)
            grid_layout = np.array([['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                                    ['H', 'I', 'J', 'K', 'L', 'M', 'N'],
                                    ['O', 'P', 'Q', 'R', 'S', 'T', 'U'],
                                    ['V', 'W', 'X', 'Y', 'Z', '1', '2'],
                                    ['3', '4', '5', '6', '7', '8', '9'],
                                    ['0', '.', ',', '?', '!', '_', 'BS']])
        else:
            # Standard 6x6 for Study D
            grid_layout = np.array([['A', 'B', 'C', 'D', 'E', 'F'],
                                    ['G', 'H', 'I', 'J', 'K', 'L'],
                                    ['M', 'N', 'O', 'P', 'Q', 'R'],
                                    ['S', 'T', 'U', 'V', 'W', 'X'],
                                    ['Y', 'Z', '1', '2', '3', '4'],
                                    ['5', '6', '7', '8', '9', '_']])
            
        decoder = P300Decoder(grid_layout)
        n_classes = decoder.n_rows * decoder.n_cols
        
        # Load MNE Epochs
        epoch_path = Path(data_dir) / study_type / f"{file_stem}-epo.fif"
        if not epoch_path.exists():
            continue
            
        epochs = mne.read_epochs(epoch_path, preload=True, verbose=False)
        X = epochs.get_data().reshape(len(epochs), -1) # Flatten for LDA
        
        # Get target probabilities for the whole session
        probs = classifier.predict_proba(X)[:, 1]
        events = epochs.events
        
        ground_truth_text = metadata['text']
        predicted_text = ""
        
        # Group flashes by character selection
        flashes_per_char = flashes_per_sequence * sequences_per_char
        
        for i in range(len(ground_truth_text)):
            start_idx = i * flashes_per_char
            end_idx = start_idx + flashes_per_char
            
            # Extract codes and probabilities for this specific character
            char_codes = events[start_idx:end_idx, 2] # Assuming StimulusCode is in the 3rd column
            char_probs = probs[start_idx:end_idx]
            
            # Decode
            pred_char = decoder.decode_character(char_codes, char_probs)
            predicted_text += pred_char
            
            total_characters += 1
            if pred_char == ground_truth_text[i]:
                correct_predictions += 1
                
        print(f"Ground Truth: {ground_truth_text}")
        print(f"Predicted:    {predicted_text}")
        
    # Final Metrics
    accuracy = correct_predictions / total_characters if total_characters > 0 else 0
    itr = decoder.calculate_itr(accuracy, n_classes=36, seconds_per_char=seconds_per_char) # Assume baseline 36 classes for ITR
    
    print("\n" + "="*30)
    print("BASELINE EVALUATION COMPLETE")
    print("="*30)
    print(f"Total Characters: {total_characters}")
    print(f"Accuracy:         {accuracy:.2%}")
    print(f"Time per Char:    {seconds_per_char:.2f} s")
    print(f"Baseline ITR:     {itr:.2f} bits/min")
    print("="*30)

if __name__ == "__main__":
    evaluate_baseline(
        data_dir="./data/processed",
        registry_path="./data/processed/ground_truth_registry.json",
        model_path="./models/calibrated_lda.joblib"
    )