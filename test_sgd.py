import numpy as np
from pathlib import Path
import gc
import mne
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mne.decoding import Vectorizer, Scaler

def test_incremental_sgd():
    processed_dir = "./data/processed"
    glob_pattern = "D_*_SE001*Train*-epo.fif"
    subject_files = sorted(Path(processed_dir).rglob(glob_pattern))[:20] # Test on 20 files
    
    print(f"Testing on {len(subject_files)} files...")
    
    sample_epochs = mne.read_epochs(subject_files[0], preload=True, verbose=False)
    info = sample_epochs.info
    scaler = Scaler(info)
    vectorizer = Vectorizer()
    std_scaler = StandardScaler()
    del sample_epochs
    gc.collect()

    base_clf = SGDClassifier(
        loss='log_loss', 
        penalty='l2', 
        alpha=0.01, 
        average=True,
        class_weight={0: 1.0, 1: 5.0},
        random_state=42
    )
    
    # Pass 1: Fit StandardScaler incrementally
    print("Fitting StandardScaler...")
    for f in subject_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data()
        X_scaled = scaler.fit_transform(X)
        X_vec = vectorizer.fit_transform(X_scaled)
        std_scaler.partial_fit(X_vec)
        del ep, X, X_scaled, X_vec
    
    # Pass 2: Fit SGD incrementally
    print("Fitting SGD...")
    for f in subject_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data()
        y = ep.events[:, 2]
        X_scaled = scaler.fit_transform(X)
        X_vec = vectorizer.fit_transform(X_scaled)
        X_std = std_scaler.transform(X_vec)
        
        base_clf.partial_fit(X_std, y, classes=[0, 1])
        del ep, X, X_scaled, X_vec, X_std
        
    print("Testing...")
    # Evaluate on the same set for a quick sanity check
    y_true = []
    y_pred = []
    for f in subject_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data()
        y = ep.events[:, 2]
        X_scaled = scaler.fit_transform(X)
        X_vec = vectorizer.fit_transform(X_scaled)
        X_std = std_scaler.transform(X_vec)
        
        probs = base_clf.predict_proba(X_std)[:, 1]
        y_true.extend(y)
        y_pred.extend(probs)
        del ep
        
    auc = roc_auc_score(y_true, y_pred)
    print(f"Train AUC: {auc:.4f}")

if __name__ == "__main__":
    test_incremental_sgd()
