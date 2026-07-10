import numpy as np
from pathlib import Path
import gc
import mne
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from mne.decoding import Vectorizer, Scaler

def load_data_in_batches(processed_dir, glob_pattern):
    """
    Iteratively loads processed .fif files and extracts lightweight Numpy arrays,
    bypassing MNE's massive memory overhead.
    """
    subject_files = sorted(Path(processed_dir).rglob(glob_pattern))
    
    if not subject_files:
        raise FileNotFoundError(f"No files found matching {glob_pattern} in {processed_dir}")
        
    print(f"Found {len(subject_files)} training runs. Streaming into NumPy arrays...")
    
    X_list = []
    y_list = []
    
    # Extract info from the first file to initialize the Scaler later
    sample_epochs = mne.read_epochs(subject_files[0], preload=True, verbose=False)
    mne_info = sample_epochs.info
    del sample_epochs
    
    for f in subject_files:
        # Load file, extract pure numbers, and immediately discard the heavy MNE object
        ep = mne.read_epochs(f, preload=True, verbose=False)
        
        X_list.append(ep.get_data())         # Shape: (Trials, Channels, Timepoints)
        y_list.append(ep.events[:, 2])       # Shape: (Trials,)
        
        # Force garbage collection to free MNE metadata memory immediately
        del ep
        
    # Stack the lightweight lists into a single continuous NumPy array
    # This will take < 200MB of RAM for the entire dataset
    X_full = np.vstack(X_list)
    y_full = np.concatenate(y_list)
    
    print(f"Dataset successfully loaded into memory.")
    print(f"Final Array Shape: {X_full.shape} | Total RAM usage: {X_full.nbytes / (1024**2):.2f} MB")
    
    return X_full, y_full, mne_info


def train_calibrated_baseline(X, y, info):
    """
    Trains a calibrated LDA classifier on raw NumPy arrays.
    """
    print("\nInitializing Classifier Pipeline...")
    
    base_clf = make_pipeline(
        Scaler(info),
        Vectorizer(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto', priors=[0.5, 0.5])
    )
    
    calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
    
    # -------------------------------------------------------------------------
    # Cross-Validation Evaluation
    # -------------------------------------------------------------------------
    print("Running 5-Fold Cross Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    brier_scores = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        calibrated_clf.fit(X_train, y_train)
        y_probs = calibrated_clf.predict_proba(X_test)[:, 1]
        
        auc_scores.append(roc_auc_score(y_test, y_probs))
        brier_scores.append(brier_score_loss(y_test, y_probs)) 
        
    print(f"Mean ROC-AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")
    print(f"Mean Brier Score: {np.mean(brier_scores):.4f} (Calibration quality)")
    
    print("\nTraining final calibrated model on full dataset...")
    calibrated_clf.fit(X, y)
    
    return calibrated_clf


if __name__ == "__main__":
    processed_dir = "./data/processed"
    
    # Target Study D, Subject 001, Training files
    # Make sure this matches your exact file naming convention!
    glob_pattern = "D_*_SE001*Train*-epo.fif"
    
    try:
        # 1. Memory-Safe Loading
        X, y, info = load_data_in_batches(processed_dir, glob_pattern)
        
        # 2. Train the baseline
        clf = train_calibrated_baseline(X, y, info)
        
        # 3. Quick sanity check on predictions
        print("\n--- Calibration Sanity Check ---")
        sample_trials = X[:5]
        probs = clf.predict_proba(sample_trials)
        print("Posterior probabilities for first 5 trials [P(Non-Target), P(Target)]:")
        for i, p in enumerate(probs):
            print(f"Trial {i+1}: [{p[0]:.4f}, {p[1]:.4f}] (True Label: {y[i]})")
        
        # 4. Save the trained model to disk
            model_save_path = f"{processed_dir}/swlda_model.pkl"
            print(f"\nSaving trained model to {model_save_path}...")
            joblib.dump(clf, model_save_path)
            print("Model saved successfully! You are ready to run the pipeline.")
            
    except Exception as e:
        print(f"Training Failed: {e}")