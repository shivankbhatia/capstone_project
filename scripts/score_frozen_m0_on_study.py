#!/usr/bin/env python3
"""Zero-shot scoring of the frozen D-trained clean M0 on a checkerboard study (E/G)."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.models.classifier_interface  # noqa: F401  (needed to unpickle the scorer)
from src.data.batch_preprocess import parse_bigp3bci_edf
from src.evaluation.sequence_scoring import aggregate_membership_logits

MAXD = 10

def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def parse(path, k):
    epochs, _, info = parse_bigp3bci_edf(
        path, tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
        baseline_correction=False, sequences_per_selection=k, verbose=False)
    md = epochs.metadata
    if md is None or not {"lit_mask", "char_index", "sequence_in_char"} <= set(md.columns):
        raise ValueError("missing group-flash/sequence metadata")
    if not info["group_flash"]:
        raise ValueError("not a group-flash run")
    return epochs, md, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--phase", choices=("Train", "Test", "both"), default="Train")
    ap.add_argument("--target-flashes-per-selection", type=int, default=20)
    ap.add_argument("--scorer", type=Path, default=ROOT / "data/processed/clean_m0_epoch_scorer.pkl")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    files = sorted(p for p in a.study_dir.rglob("*.edf") if not p.name.startswith("._")
                   and (a.phase == "both" or f"/{a.phase}/" in p.as_posix()))
    if a.max_runs:
        files = files[:a.max_runs]
    scorer = None if a.audit_only else joblib.load(a.scorer)
    runs, failures = [], []
    subj_logits, subj_y = {}, {}
    chars = {}            # subject -> {depth: [n, correct]}
    invalid = 0
    for i, path in enumerate(files, 1):
        run_id = path.stem
        subject = run_id.split("_SE", 1)[0]
        try:
            epochs, md, info = parse(path, a.target_flashes_per_selection)
            rec = {"run_id": run_id, "epochs": len(epochs), "flashes_per_seq": int(info["flashes_per_seq"]),
                   "group_flash": bool(info["group_flash"]), "channels": len(epochs.ch_names),
                   "characters": int(md["char_index"].max()) + 1 if len(md) else 0}
            if not a.audit_only:
                want = list(scorer.channels)
                if set(want) != set(epochs.ch_names):
                    raise ValueError(f"channel mismatch: {sorted(set(want) ^ set(epochs.ch_names))}")
                epochs.reorder_channels(want)
                X = epochs.get_data(copy=False).astype(np.float32)
                y = epochs.events[:, 2].astype(int)
                masks = np.asarray([[int(b) for b in m] for m in md["lit_mask"].astype(str)], dtype=np.uint8)
                if masks.shape[1] != 72:
                    raise ValueError(f"expected 72-cell masks, got {masks.shape}")
                raw = scorer.decision_function(X)
                logits = scorer.calibrated_logits(X)
                subj_logits.setdefault(subject, []).append(raw)
                subj_y.setdefault(subject, []).append(y)
                char = md["char_index"].to_numpy(int)
                seq = md["sequence_in_char"].to_numpy(int)
                for c in np.unique(char[char >= 0]):
                    idx = np.flatnonzero(char == c)
                    tm = masks[idx[y[idx] == 1]]
                    inter = np.all(tm.astype(bool), axis=0) if len(tm) else np.zeros(72, bool)
                    if inter.sum() != 1:
                        invalid += 1
                        continue
                    truth = int(np.flatnonzero(inter)[0])
                    cum = np.zeros(72)
                    for depth, s in enumerate(np.sort(np.unique(seq[idx][seq[idx] >= 0])), 1):
                        loc = idx[seq[idx] == s]
                        post = aggregate_membership_logits(logits[loc], masks[loc], n_rows=9, n_cols=8)
                        cum += np.log(np.clip(post, 1e-12, 1.0))
                        if depth <= MAXD:
                            d = chars.setdefault(subject, {}).setdefault(depth, [0, 0])
                            d[0] += 1
                            d[1] += int(softmax(cum).argmax() == truth)
            runs.append(rec)
        except Exception as exc:  # record and continue
            failures.append({"file": str(path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", flush=True)
        if i % 10 == 0 or i == len(files):
            print(f"{i}/{len(files)} files, {len(failures)} failures", flush=True)

    out = {"label": a.label, "phase": a.phase, "files": len(files), "audit_only": a.audit_only,
           "target_flashes_per_selection": a.target_flashes_per_selection,
           "scorer": None if a.audit_only else str(a.scorer),
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip(),
           "runs": runs, "failures": failures}
    if not a.audit_only and not subj_logits:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        raise SystemExit(f"No run was scored; first error: {failures[0]['error'] if failures else 'none'}")
    if not a.audit_only:
        auc = {}
        for s in subj_logits:
            yy, ll = np.concatenate(subj_y[s]), np.concatenate(subj_logits[s])
            if len(set(yy)) == 2:
                auc[s] = float(roc_auc_score(yy, ll))
        yy = np.concatenate([np.concatenate(v) for v in subj_y.values()])
        ll = np.concatenate([np.concatenate(v) for v in subj_logits.values()])
        out["epoch_auc_pooled"] = float(roc_auc_score(yy, ll))
        out["epoch_auc_per_subject"] = auc
        out["invalid_target_intersections"] = invalid
        depths = {}
        for s, dd in chars.items():
            for depth, (n, c) in dd.items():
                depths.setdefault(depth, [0, 0])
                depths[depth][0] += n
                depths[depth][1] += c
        out["accuracy_by_depth"] = {str(d): {"characters": n, "accuracy": c / n}
                                    for d, (n, c) in sorted(depths.items())}
        out["character_accuracy_per_subject_at_max_depth"] = {
            s: {"characters": dd[max(dd)][0], "accuracy": dd[max(dd)][1] / dd[max(dd)][0]}
            for s, dd in sorted(chars.items())}
    args_out = a.output
    args_out.parent.mkdir(parents=True, exist_ok=True)
    args_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args_out}")

if __name__ == "__main__":
    main()