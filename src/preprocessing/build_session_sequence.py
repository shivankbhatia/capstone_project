import json
import os

def yield_character_trials(registry_path="data/processed/ground_truth_registry.json", dataset_dir=None):
    """
    Yields character trials for the end-to-end replay loop.
    
    If the processed registry exists, it yields real dataset targets.
    If not, it falls back to a mock sequence so the Day 8 ablation 
    pipeline can be tested immediately.
    
    Args:
        registry_path: Path to the JSON registry of sessions.
        dataset_dir: Optional base directory for the dataset.
        
    Yields:
        dict: Containing 'target_char' and 'context_so_far' (previously spelled chars)
    """
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
        
        total_yielded = 0
        
        if isinstance(registry, dict):
            # FIX: Iterate directly over the key-value pairs
            # Key = session_id (e.g., "D_15_SE001_Dyn_Test01")
            # Value = target_word (e.g., "C")
            for session_id, target_word in registry.items():
                if not isinstance(target_word, str):
                    continue
                
                # Reconstruct the filename based on your screenshot structure
                file_name = f"{session_id}-epo.fif"

                studyd_path = os.path.join(dataset_dir, "StudyD", file_name)
                if os.path.exists(studyd_path):
                    file_path = studyd_path
                else:
                    # If it's not in StudyD, we skip this character trial
                    continue 
                
                # Attempt to build the full path if dataset_dir is provided
                file_path = file_name
                if dataset_dir:
                    # Check inside StudyD subfolder just in case
                    studyd_path = os.path.join(dataset_dir, "StudyD", file_name)
                    if os.path.exists(studyd_path):
                        file_path = studyd_path
                    else:
                        file_path = os.path.join(dataset_dir, file_name)
                
                for i, char in enumerate(target_word):
                    total_yielded += 1
                    yield {
                        'target_char': char,
                        'context_so_far': target_word[:i],
                        'eeg_data_path': file_path
                    }
        else:
            print(f"⚠️ Warning: Unexpected JSON structure in {registry_path}. Expected a dictionary.")
            
        if total_yielded == 0:
            print(f"\n⚠️ ERROR: Found registry at {registry_path} but extracted 0 characters!")
            print(f"⚠️ Check if the JSON is completely empty.\n")
            
    else:
        # MOCK MODE: Ensures the pipeline runs even if data isn't fully preprocessed yet
        print("⚠️ ground_truth_registry.json not found. Using MOCK sequence for pipeline testing.")
        mock_words = ["WATER", "HELLO", "YES"]
        for word in mock_words:
            for i, char in enumerate(word):
                yield {
                    'target_char': char,
                    'context_so_far': word[:i],
                    'eeg_data_path': 'mock_path'
                }