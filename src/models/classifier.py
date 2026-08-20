import numpy as np
from pathlib import Path
import gc
import mne
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer, Scaler

def train_calibrated_batch(processed_dir, glob_pattern, epochs=5):
    """
    Trains an SGDClassifier (logistic regression) file-by-file to avoid
    loading multiple files or the entire dataset into memory at once.
    """
    subject_files = sorted(Path(processed_dir).rglob(glob_pattern))
    
    if not subject_files:
        raise FileNotFoundError(f"No files found matching {glob_pattern} in {processed_dir}")
        
    print(f"Found {len(subject_files)} training runs. Starting incremental batch training...")
    
    # Initialize Scaler and Vectorizer using the first file's info
    sample_epochs = mne.read_epochs(subject_files[0], preload=True, verbose=False)
    info = sample_epochs.info
    scaler = Scaler(info)
    vectorizer = Vectorizer()
    std_scaler = StandardScaler()
    del sample_epochs
    gc.collect()

    print("Pass 1: Incrementally fitting StandardScaler (required for SGD convergence)...")
    for i, f in enumerate(subject_files):
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data(copy=False)
        X_scaled = scaler.fit_transform(X)
        X_vec = vectorizer.fit_transform(X_scaled)
        std_scaler.partial_fit(X_vec)
        if (i + 1) % 20 == 0 or (i + 1) == len(subject_files):
            print(f"  Processed {i + 1}/{len(subject_files)} files for scaling...")
        del ep, X, X_scaled, X_vec
        gc.collect()

    # SGDClassifier with log_loss outputs probabilistic predictions natively.
    # Averaged SGD and class weights handle the noisy, imbalanced P300 data.
    base_clf = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        alpha=0.01, 
        average=True,
        class_weight={0: 1.0, 1: 5.0},
        random_state=42
    )
    
    print(f"\nPass 2: Incrementally training SGDClassifier for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"--- Epoch {epoch + 1}/{epochs} ---")
        for i, f in enumerate(subject_files):
            ep = mne.read_epochs(f, preload=True, verbose=False)
            X = ep.get_data(copy=False)
            y = ep.events[:, 2]
            
            X_scaled = scaler.fit_transform(X)
            X_vec = vectorizer.fit_transform(X_scaled)
            X_std = std_scaler.transform(X_vec)
            
            base_clf.partial_fit(X_std, y, classes=[0, 1])
                
            del ep, X, X_scaled, X_vec, X_std
            
        print(f"  Epoch {epoch + 1} complete.")
        gc.collect()

    # Wrap the transformers and base classifier in a pipeline
    pipeline_clf = make_pipeline(
        scaler,
        vectorizer,
        std_scaler,
        base_clf
    )
    
    return pipeline_clf

if __name__ == "__main__":
    processed_dir = "./data/processed"
    glob_pattern = "D_*_SE001*Train*-epo.fif"
    
    try:
        clf = train_calibrated_batch(processed_dir, glob_pattern, epochs=5)
        
        # Save the trained model to disk
        model_save_path = f"{processed_dir}/swlda_model.pkl"
        print(f"\nSaving trained model to {model_save_path}...")
        joblib.dump(clf, model_save_path)
        print("Model saved successfully! You are ready to run the pipeline.")
            
    except Exception as e:
        print(f"Training Failed: {e}")