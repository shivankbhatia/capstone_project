import json
import traceback
from pathlib import Path

# Absolute import matching your repository structure
from src.preprocessing.epoching import parse_bigp3bci_edf

def batch_process_studies(raw_dirs, output_dir):
    """
    Recursively finds .edf files in given directories, processes them into MNE epochs,
    and maintains a JSON registry of the ground-truth text.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry_path = output_dir / "ground_truth_registry.json"
    
    if registry_path.exists():
        with open(registry_path, "r") as f:
            registry = json.load(f)
    else:
        registry = {}

    edf_files = []
    for directory in raw_dirs:
        dir_path = Path(directory)
        if dir_path.exists():
            edf_files.extend(list(dir_path.rglob("*.edf")))
        else:
            print(f"Warning: Directory not found - {dir_path}")

    print(f"Found {len(edf_files)} .edf files to process across selected studies.")

    for edf_path in edf_files:
        file_stem = edf_path.stem 
        
        # Extract the Study folder name (e.g., 'StudyD') from the raw path
        # If it somehow fails, it falls back to parsing the first letter of the file (e.g., 'D' -> 'StudyD')
        study_folder_name = next(
            (part for part in edf_path.parts if part.startswith("Study")), 
            f"Study{file_stem.split('_')[0]}"
        )
        
        # Create the study-specific subdirectory inside /processed/
        study_out_dir = output_dir / study_folder_name
        study_out_dir.mkdir(parents=True, exist_ok=True)
        
        output_fif = study_out_dir / f"{file_stem}-epo.fif"

        if output_fif.exists():
            print(f"Skipping {file_stem} — already processed.")
            continue

        print(f"\n{'-'*60}\nProcessing: {file_stem} -> {study_folder_name}\n{'-'*60}")
        
        try:
            epochs, spelled_string, grid_map = parse_bigp3bci_edf(str(edf_path))
            
            # Downsample to 20Hz BEFORE saving to disk to prevent memory overflow downstream
            print("Downsampling to 20Hz to permanently shrink footprint...")
            epochs.resample(20, verbose=False)
            
            epochs.save(output_fif, overwrite=True)
            
            registry[file_stem] = spelled_string
            with open(registry_path, "w") as f:
                json.dump(registry, f, indent=4)
                
            print(f"Successfully saved {output_fif.name} into {study_folder_name}/")

        except Exception as e:
            print(f"FAILED processing {file_stem}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    raw_study_d = "./data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyD"
    raw_study_e = "./data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyE"
    
    out_dir = "./data/processed"
    
    batch_process_studies(
        raw_dirs=[raw_study_d, raw_study_e], 
        output_dir=out_dir
    )