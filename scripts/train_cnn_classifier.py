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
import sys
from pathlib import Path

# Running this file directly (`python scripts/train_cnn_classifier.py`) only
# puts scripts/ on sys.path, not the repo root -- so `from src...` below
# fails with ModuleNotFoundError regardless of cwd. Insert the repo root
# (parent of scripts/) explicitly so the script works exactly as documented
# above, run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def run_epoch(model, loader, criterion, optimizer, device, train=True,
              grad_clip_norm=1.0):
    model.train(mode=train)
    total_loss, n_total = 0.0, 0
    # Per-class correct/total for balanced accuracy -- raw accuracy under
    # ~88%/12% imbalance is dominated by the majority class and can't tell
    # a model that's actually learning P300 structure apart from one that
    # has collapsed to "always predict non-target".
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    for rp_img, cwt_img, labels in loader:
        rp_img, cwt_img, labels = rp_img.to(device), cwt_img.to(device), labels.to(device)

        with torch.set_grad_enabled(train):
            logits = model(rp_img, cwt_img)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                # Caps the gradient norm so a batch with a few confidently-
                # wrong positives (loss amplified by pos_weight) can't blow
                # up the update -- this is the direct fix for loss spiking
                # to ~21 during training.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        for c in (0, 1):
            mask = labels == c
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
        n_total += labels.size(0)

    raw_acc = sum(class_correct.values()) / max(1, n_total)
    per_class_recall = {
        c: class_correct[c] / class_total[c] if class_total[c] > 0 else float("nan")
        for c in (0, 1)
    }
    balanced_acc = np.nanmean([per_class_recall[0], per_class_recall[1]])

    return total_loss / max(1, n_total), raw_acc, balanced_acc, per_class_recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    # 1e-3 was too high for this architecture (16k-dim fused Linear head,
    # default init) and was the main driver of the loss-spike instability;
    # 1e-4 + grad clipping + ReduceLROnPlateau below is the stable regime.
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=3.0,
                         help="Class weight for the target class. Lowered from "
                              "5.0 -- combined with the old 1e-3 LR and no grad "
                              "clipping, 5.0 let a single confidently-wrong "
                              "positive prediction dominate the batch loss.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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

    # Majority-class baseline, printed once, so val_acc can be read in
    # context -- ~88% raw accuracy is meaningless on its own if the
    # majority class alone gets you there.
    train_labels = np.concatenate([
        h5py.File(p, "r")["labels"][:] for p in train_files
    ])
    majority_frac = max(train_labels.mean(), 1 - train_labels.mean())
    print(f"Majority-class (always predict non-target) baseline accuracy: "
          f"{majority_frac:.4f}  (target class prevalence: {train_labels.mean():.4f})")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Selecting the checkpoint on val_loss alone is exactly what let a
    # majority-class-collapsed model look "best": val_loss can improve while
    # the model still never learns to recognize the target class. Balanced
    # accuracy (mean of per-class recall) can't be gamed that way.
    best_balanced_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_bal_acc, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_loss, val_acc, val_bal_acc, val_recall = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False,
        )
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_bal_acc={train_bal_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal_acc:.4f}  "
              f"val_recall[non-target]={val_recall[0]:.4f} val_recall[target]={val_recall[1]:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_bal_acc > best_balanced_acc:
            best_balanced_acc = val_bal_acc
            CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_balanced_acc": val_bal_acc,
                "epoch": epoch,
            }, CKPT_PATH)
            print(f"  -> saved checkpoint ({CKPT_PATH}) [best balanced_acc={best_balanced_acc:.4f}]")

    with open(FEATURES_DIR.parent / "cnn_test_sessions.json", "w") as f:
        json.dump([f.stem.replace("-epo-feat", "") for f in test_files], f, indent=2)

    print(f"\nDone. Best val_balanced_acc={best_balanced_acc:.4f} "
          f"(majority-class baseline was {majority_frac:.4f}). "
          f"Test-session list saved for evaluation wiring into run_pipeline.py.")