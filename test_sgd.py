import numpy as np
from pathlib import Path
import gc
import mne
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from mne.decoding import Vectorizer, Scaler

def test_sgd():
    processed_dir = "./data/processed"
    glob_pattern = "D_*_SE001*Train*-epo.fif"
    subject_files = sorted(Path(processed_dir).rglob(glob_pattern))[:15] # Subset for testing
    
    scaler = Scaler(mne.read_epochs(subject_files[0], preload=False).info)
    vectorizer = Vectorizer()
    std_scaler = StandardScaler()
    
    print("Pass 1: Fitting StandardScaler incrementally...")
    for f in subject_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data(copy=False)
        X_vec = vectorizer.fit_transform(scaler.fit_transform(X))
        std_scaler.partial_fit(X_vec)
        del ep, X, X_vec
        
    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, 
                        average=True, class_weight={0:1.0, 1:5.0}, random_state=42)
    
    epochs = 5
    print(f"Pass 2: Training SGD for {epochs} epochs...")
    for epoch in range(epochs):
        for f in subject_files:
            ep = mne.read_epochs(f, preload=True, verbose=False)
            X = ep.get_data(copy=False)
            y = ep.events[:, 2]
            X_vec = std_scaler.transform(vectorizer.fit_transform(scaler.fit_transform(X)))
            clf.partial_fit(X_vec, y, classes=[0, 1])
            del ep, X, X_vec
        print(f"  Finished epoch {epoch+1}")
        
    print("Evaluating...")
    y_true, y_pred = [], []
    for f in subject_files:
        ep = mne.read_epochs(f, preload=True, verbose=False)
        X = ep.get_data(copy=False)
        y = ep.events[:, 2]
        X_vec = std_scaler.transform(vectorizer.fit_transform(scaler.fit_transform(X)))
        y_true.extend(y)
        y_pred.extend(clf.predict_proba(X_vec)[:, 1])
        del ep, X, X_vec
        
    auc = roc_auc_score(y_true, y_pred)
    print(f"AUC after {epochs} epochs: {auc:.4f}")

if __name__ == '__main__':
    test_sgd()
