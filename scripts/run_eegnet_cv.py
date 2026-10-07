#!/usr/bin/env python3
"""Run grouped OOF EEGNet with flash BCE / sequence-loss ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_shrinkage_lda import _character_metrics, _load_metadata  # noqa: E402
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint, load_heldout_run_ids,
)
from src.models.eegnet import EEGNetClassifier  # noqa: E402


def _load_sequences(h5, meta, cfg, channels):
    channel_idx = list(range(len(channels))) if cfg["channels"] == "all" else [
        channels.index(c) for c in ("Fz", "Cz", "Pz", "PO7", "PO8", "Oz")
    ]
    sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
    times = tmin + np.arange(h5["X"].shape[-1]) / sfreq
    time_keep = (times >= 0) & (times <= cfg["window_s"] + 1e-8)
    groups = {}
    for i, (run, char_idx, seq_idx) in enumerate(
        zip(meta["run_id"], meta["char_idx"], meta["seq_idx"])
    ):
        if char_idx >= 0 and seq_idx >= 0:
            groups.setdefault((str(run), int(char_idx), int(seq_idx)), []).append(i)
    keys, row_groups = [], []
    for key, rows in sorted(groups.items()):
        if len(rows) != 17:
            continue
        keys.append(key)
        row_groups.append(np.asarray(rows, dtype=int))
    if not keys:
        raise ValueError("No complete 17-flash sequences found in cache")
    flat_idx = np.concatenate(row_groups)
    X = h5["X"][flat_idx][:, channel_idx, :][:, :, time_keep][:, :, ::int(cfg["decim"])]
    X = X.reshape(len(keys), 17, len(channel_idx), -1)
    y = meta["y"][flat_idx].reshape(len(keys), 17)
    codes = meta["stim_code"][flat_idx].reshape(len(keys), 17)
    subjects = np.asarray([key[0].split("_SE", 1)[0] for key in keys])
    run_ids = np.asarray([key[0] for key in keys])
    row_groups = np.stack(row_groups)
    return X, y, codes, subjects, run_ids, row_groups


def _metrics(scores, y, meta):
    valid = np.isfinite(scores)
    flat_y, flat_scores = y.reshape(-1), scores.reshape(-1)
    mask = np.isfinite(flat_scores)
    # Remove any padded/missing scores; complete sequences are kept.
    row_idx = meta["row_groups"].reshape(-1)
    return {
        "oof_epochs": int(mask.sum()),
        "epoch_auc": float(roc_auc_score(flat_y[mask], flat_scores[mask])),
        "character_metrics": _character_metrics(
            flat_scores[mask], flat_y[mask], meta["run_id_rows"][mask],
            meta["char_idx_rows"][mask], meta["seq_idx_rows"][mask],
            meta["stim_code_rows"][mask],
        ),
    }


def run(cache_path, config_path, manifest_path, folds, seeds, modes, scores_output=None,
        epochs=30, torch_threads=1):
    torch.set_num_threads(torch_threads)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    heldout = load_heldout_run_ids(manifest_path)
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        channels = str(h5.attrs["channels"]).split(",")
        X, y_seq, codes, subjects, run_ids, row_groups = _load_sequences(h5, meta, cfg, channels)
        assert_training_pool_disjoint(
            [f"{run_id}-epo.fif" for run_id in sorted(set(run_ids))], heldout
        )
        n_folds = min(folds, len(set(run_ids)))
        splitter = GroupKFold(n_splits=n_folds)
        flat_metadata = {
            "row_groups": row_groups,
            "run_id_rows": meta["run_id"][row_groups.reshape(-1)],
            "char_idx_rows": meta["char_idx"][row_groups.reshape(-1)],
            "seq_idx_rows": meta["seq_idx"][row_groups.reshape(-1)],
            "stim_code_rows": meta["stim_code"][row_groups.reshape(-1)],
        }
        model_results = {}
        score_arrays = {}
        for mode in modes:
            seed_scores = []
            for seed in seeds:
                oof = np.full(y_seq.shape, np.nan, dtype=np.float32)
                for fold, (train_idx, val_idx) in enumerate(
                    splitter.split(X, y_seq[:, 0], groups=run_ids), start=1
                ):
                    model = EEGNetClassifier(seed=seed + fold, epochs=epochs, patience=6)
                    model.fit_sequence(
                        X[train_idx], y_seq[train_idx], codes[train_idx],
                        run_ids[train_idx], loss_mode=mode,
                    )
                    val_logits = model.decision_function(X[val_idx].reshape(-1, X.shape[2], X.shape[3]))
                    oof[val_idx] = val_logits.reshape(len(val_idx), 17)
                    print(f"EEGNet {mode} seed {seed} fold {fold}/{n_folds} ({model.device})", flush=True)
                seed_scores.append(oof)
            mean_scores = np.nanmean(np.stack(seed_scores), axis=0)
            score_arrays[mode] = mean_scores.reshape(-1)
            flat_scores = mean_scores.reshape(-1)
            row_idx = row_groups.reshape(-1)
            row_mask = np.isfinite(flat_scores)
            model_results[mode] = {
                "seeds": seeds,
                "device": "mps" if torch.backends.mps.is_available() else "cpu",
                "epoch_auc": float(roc_auc_score(
                    meta["y"][row_idx[row_mask]], flat_scores[row_mask]
                )),
                "character_metrics": _character_metrics(
                    flat_scores[row_mask], meta["y"][row_idx[row_mask]],
                    meta["run_id"][row_idx[row_mask]], meta["char_idx"][row_idx[row_mask]],
                    meta["seq_idx"][row_idx[row_mask]], meta["stim_code"][row_idx[row_mask]],
                ),
            }

    if scores_output is not None:
        scores_output.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = row_groups.reshape(-1)
        payload = {
            "y": meta["y"][flat_rows],
            "subject": meta["subject"][flat_rows],
            "run_id": meta["run_id"][flat_rows],
            "char_idx": meta["char_idx"][flat_rows],
            "seq_idx": meta["seq_idx"][flat_rows],
            "stim_code": meta["stim_code"][flat_rows],
        }
        payload.update({f"logits_{name}": value for name, value in score_arrays.items()})
        np.savez_compressed(scores_output, **payload)
    return {
        "cache": str(cache_path),
        "preprocessing": cfg,
        "cv": {"method": "GroupKFold", "group": "run_id", "folds": n_folds},
        "seeds": seeds,
        "models": model_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--mode", action="append", choices=("bce", "softmax", "both"),
                        default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_eegnet_inner.json")
    parser.add_argument("--scores-output", type=Path,
                        default=ROOT / "results/tables/classifier_eegnet_inner_scores.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    modes = args.mode or ["bce", "softmax", "both"]
    result = run(args.cache, args.config, args.manifest, args.folds, args.seeds, modes,
                 args.scores_output, epochs=args.epochs, torch_threads=args.torch_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(args.output.read_text()) if args.output.exists() else {}
    all_results[args.config.stem] = result
    args.output.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({mode: {"epoch_auc": value["epoch_auc"],
                             "char_acc": value["character_metrics"]["full_repetition_accuracy"]}
                      for mode, value in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
