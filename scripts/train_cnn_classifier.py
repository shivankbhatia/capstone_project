"""Train P300CNN (src/models/cnn_classifier.py) on RP + CWT features.

Reads data/processed/StudyD/features/*-feat.h5 (produced by
scripts/compute_2d_features.py). Splits by SESSION, not by epoch, to avoid
leaking epochs from the same session across train/val/test -- consistent
with the leakage discipline already used for the RAG phrase bank
(data/processed/test_sessions.json holds out the same 20% test sessions,
so this CNN and the existing RAG-LM ablation arm are evaluated on the
identical held-out split).

Usage: python scripts/train_cnn_classifier.py [--epochs 15] [--batch_size 64]
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.cnn_classifier import P300CNN, prepare_batch

FEATURES_DIR = Path("data/processed/StudyD/features")
TEST_SPLIT_PATH = Path("data/processed/test_sessions.json")
CKPT_PATH = Path("data/processed/cnn_p300_model.pt")


class FeatureFileDataset(Dataset):
    """One (rp_triu, cwt, label) sample per epoch, lazily read from HDF5."""

    def __init__(self, h5_paths):
        self.index = []  # (file_path, epoch_idx)
        self._cache = {}
        for p in h5_paths:
            with h5py.File(p, "r") as h5f:
                n = h5f.attrs["n_epochs"]
            for i in range(n):
                self.index.append((p, i))

    def __len__(self):
        return len(self.index)

    def _get_file(self, path):
        if path not in self._cache:
            # rdcc_nbytes caps per-file HDF5 chunk cache at 8MB (default is
            # often larger); with num_workers>0 each worker holds its own
            # cache across all files it touches, so this matters on 16GB.
            self._cache[path] = h5py.File(path, "r", rdcc_nbytes=8 * 1024 * 1024)
        return self._cache[path]

    def __getitem__(self, idx):
        path, i = self.index[idx]
        h5f = self._get_file(path)
        rp = h5f["recurrence_2d"][i]      # (C, triu_len)
        cwt = h5f["cwt_2d"][i]            # (C, F, T)
        label = h5f["labels"][i]
        return rp, cwt, label


def collate(batch):
    rps, cwts, labels = zip(*batch)
    rp_arr = np.stack(rps)
    cwt_arr = np.stack(cwts)
    label_arr = np.array(labels, dtype=np.float32)
    rp_img, cwt_img = prepare_batch(rp_arr, cwt_arr)
    return rp_img, cwt_img, torch.from_numpy(label_arr)


def session_split(all_files, test_sessions, val_fraction=0.15, seed=42):
    """Train/val/test split by session, held-out test matches the RAG split."""
    test_files, trainval_files = [], []
    for f in all_files:
        session_id = f.stem.replace("-epo-feat", "")
        (test_files if session_id in test_sessions else trainval_files).append(f)

    rng = np.random.RandomState(seed)
    trainval_files = list(trainval_files)
    rng.shuffle(trainval_files)
    n_val = max(1, int(len(trainval_files) * val_fraction))
    val_files = trainval_files[:n_val]
    train_files = trainval_files[n_val:]
    return train_files, val_files, test_files


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(mode=train)
    total_loss, n_correct, n_total = 0.0, 0, 0
    for rp_img, cwt_img, labels in loader:
        rp_img, cwt_img, labels = rp_img.to(device), cwt_img.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(rp_img, cwt_img)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

    return total_loss / max(1, n_total), n_correct / max(1, n_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos_weight", type=float, default=5.0,
                         help="Class weight for the target class (matches "
                              "classifier.py's 0:1.0, 1:5.0 imbalance handling).")
    parser.add_argument("--num_workers", type=int, default=2,
                         help="DataLoader worker processes for parallel HDF5 reads. "
                              "Kept modest (2) by default for 16GB machines -- each "
                              "worker holds its own HDF5 handle cache; set 0 to "
                              "disable multiprocessing entirely if RAM is tight.")
    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    all_files = sorted(FEATURES_DIR.glob("*-feat.h5"))
    if not all_files:
        raise FileNotFoundError(
            f"No feature files under {FEATURES_DIR} -- run "
            f"scripts/compute_2d_features.py first."
        )

    test_sessions = set()
    if TEST_SPLIT_PATH.exists():
        test_sessions = set(json.load(open(TEST_SPLIT_PATH)))
    else:
        print("WARNING: test_sessions.json not found -- CNN test split will "
              "not match the RAG-LM held-out split. Run "
              "scripts/build_phrase_bank_from_registry.py first for a fair "
              "comparison across ablation arms.")

    train_files, val_files, test_files = session_split(all_files, test_sessions)
    print(f"Sessions -> train: {len(train_files)}, val: {len(val_files)}, "
          f"test: {len(test_files)}")

    train_ds = FeatureFileDataset(train_files)
    val_ds = FeatureFileDataset(val_files)
    print(f"Epochs -> train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, num_workers=args.num_workers,
                               persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, num_workers=args.num_workers,
                             persistent_workers=args.num_workers > 0)

    model = P300CNN(n_channels=8).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
            }, CKPT_PATH)
            print(f"  -> saved checkpoint ({CKPT_PATH})")

    with open(FEATURES_DIR.parent / "cnn_test_sessions.json", "w") as f:
        json.dump([f.stem.replace("-epo-feat", "") for f in test_files], f, indent=2)

    print(f"\nDone. Best val_loss={best_val_loss:.4f}. "
          f"Test-session list saved for evaluation wiring into run_pipeline.py.")