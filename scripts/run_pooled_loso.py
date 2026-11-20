#!/usr/bin/env python3
"""Pooled leave-one-subject-out evaluation of the clean M0 recipe across Studies D, E, G (16 ch).

Exploratory. D cache must be the clean 16-channel RC/Train cache; E/G use Train files only.
Accuracy is uncalibrated full-repetition accuracy (no temperature, no stopping).
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import h5py, numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from scripts.run_study_d_classifier_loso import _character_metrics as d_metrics  # noqa: E402
from scripts.screen_shrinkage_lda import _load_metadata  # noqa: E402
from src.data.batch_preprocess import parse_bigp3bci_edf  # noqa: E402
from src.evaluation.sequence_scoring import aggregate_membership_logits  # noqa: E402

CONFIG = ROOT / "configs/classifier_optimization/preprocessing/baseline_off.json"
MAXD, CELLS = 10, 72


def build_pooled_cache(d_cache, e_dir, g_dir, out):
    if out.exists():
        return
    sd = h5py.string_dtype()
    with h5py.File(d_cache, "r") as src, h5py.File(out, "w") as dst:
        channels = str(src.attrs["channels"]).split(",")
        sfreq, tmin = float(src.attrs["sfreq"]), float(src.attrs["tmin"])
        n_d, c, t = src["X"].shape
        dX = dst.create_dataset("X", shape=(n_d, c, t), maxshape=(None, c, t), dtype="f4", chunks=(64, c, t))
        for s in range(0, n_d, 4096):
            dX[s:s + 4096] = src["X"][s:s + 4096]
        d = _load_metadata(src)
        cols = {k: [np.asarray(d[k])] for k in ("y", "subject", "run_id", "char_idx", "seq_idx", "stim_code")}
        cols["study"] = [np.array(["D"] * n_d, dtype=object)]
        masks = [np.zeros((n_d, CELLS), np.uint8)]
        for study, root in (("E", e_dir), ("G", g_dir)):
            files = sorted(p for p in root.rglob("*.edf")
                           if not p.name.startswith("._") and "/Train/" in p.as_posix())
            for i, path in enumerate(files, 1):
                ep, _, info = parse_bigp3bci_edf(path, tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
                                                 baseline_correction=False, verbose=False)
                md = ep.metadata
                if not info["group_flash"] or md is None:
                    raise ValueError(f"{path.name}: not a group-flash run")
                if set(ep.ch_names) != set(channels):
                    raise ValueError(f"{path.name}: channel mismatch")
                ep.reorder_channels(channels)
                if abs(ep.info["sfreq"] - sfreq) > 1e-6 or abs(ep.tmin - tmin) > 1e-6:
                    raise ValueError(f"{path.name}: sfreq/tmin differ from D cache")
                X = ep.get_data(copy=False).astype(np.float32)
                if X.shape[1:] != (c, t):
                    raise ValueError(f"{path.name}: epoch shape {X.shape[1:]} != {(c, t)}")
                n = len(X)
                dX.resize(dX.shape[0] + n, axis=0)
                dX[-n:] = X
                run_id = path.stem
                cols["y"].append(ep.events[:, 2].astype(np.uint8))
                cols["subject"].append(np.array([run_id.split("_SE", 1)[0]] * n, dtype=object))
                cols["run_id"].append(np.array([run_id] * n, dtype=object))
                cols["char_idx"].append(md["char_index"].to_numpy(np.int32))
                cols["seq_idx"].append(md["sequence_in_char"].to_numpy(np.int32))
                cols["stim_code"].append(np.full(n, -1, np.int32))
                cols["study"].append(np.array([study] * n, dtype=object))
                masks.append(np.asarray([[int(b) for b in m] for m in md["lit_mask"].astype(str)], np.uint8))
                if i % 10 == 0:
                    print(f"cache {study}: {i}/{len(files)}", flush=True)
        for k, parts in cols.items():
            arr = np.concatenate(parts)
            dst.create_dataset(k, data=arr, dtype=sd) if arr.dtype == object else dst.create_dataset(k, data=arr)
        dst.create_dataset("lit_mask", data=np.concatenate(masks))
        for k, v in (("channels", ",".join(channels)), ("sfreq", sfreq), ("tmin", tmin)):
            dst.attrs[k] = v


def group_metrics(logits, y, run, char, seq, masks):
    groups = {}
    for i, (r, c) in enumerate(zip(run, char)):
        if c >= 0:
            groups.setdefault((r, int(c)), []).append(i)
    depth, finals, invalid = {d: [0, 0] for d in range(1, MAXD + 1)}, [], 0
    for rows in groups.values():
        idx = np.asarray(rows)
        tm = masks[idx[y[idx] == 1]].astype(bool)
        inter = tm.all(axis=0) if len(tm) else np.zeros(CELLS, bool)
        if inter.sum() != 1:
            invalid += 1
            continue
        truth, cum, last = int(np.flatnonzero(inter)[0]), np.zeros(CELLS), None
        for d, s in enumerate(np.unique(seq[idx][seq[idx] >= 0])[:MAXD], 1):
            loc = idx[seq[idx] == s]
            cum += np.log(np.clip(aggregate_membership_logits(logits[loc], masks[loc], 9, 8), 1e-12, 1.0))
            last = int(cum.argmax() == truth)
            depth[d][0] += 1
            depth[d][1] += last
        if last is not None:
            finals.append(last)
    return {"characters": len(finals), "full_accuracy": float(np.mean(finals)) if finals else None,
            "invalid_targets": invalid,
            "accuracy_by_depth": {str(d): (c / n if n else None) for d, (n, c) in depth.items()}}


def evaluate(m, study, idx, scores):
    s, y = scores[idx], m["y"][idx]
    out = {"epochs": int(len(idx)), "auc": float(roc_auc_score(y, s)) if len(set(y)) == 2 else None}
    if study == "D":
        meta = {k: m[k][idx] for k in ("run_id", "char_idx", "seq_idx", "stim_code")}
        r = d_metrics(s, y, meta)["overall"]
        out.update(characters=r["characters"], full_accuracy=r["full_repetition_accuracy"])
    else:
        out.update(group_metrics(s, y, m["run_id"][idx], m["char_idx"][idx], m["seq_idx"][idx],
                                 m["lit_mask"][idx]))
    return out


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare(a, b, seed=0, n_boot=10000):
    d = np.asarray(a, float) - np.asarray(b, float)
    boots = np.random.default_rng(seed).choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    p = float(wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
    return {"n_subjects": len(d), "mean_difference": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p, "better": int((d > 0).sum()), "tied": int((d == 0).sum()), "worse": int((d < 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline_16ch.h5")
    ap.add_argument("--e-dir", type=Path, required=True)
    ap.add_argument("--g-dir", type=Path, required=True)
    ap.add_argument("--pooled-cache", type=Path, default=ROOT / "data/cache/pooled_DEG_16ch.h5")
    ap.add_argument("--arms", default="pooled,d_only,own_study")
    ap.add_argument("--limit-subjects", type=int, help="smoke test: held-out subjects per study")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed-offset", type=int, default=0)   # add next to the other arguments
    a = ap.parse_args()
    arms = a.arms.split(",")
    build_pooled_cache(a.d_cache, a.e_dir, a.g_dir, a.pooled_cache)
    cfg = json.loads(CONFIG.read_text())
    results = json.loads(a.output.read_text())["results"] if a.output.exists() else {}

    def save(summary=None):
        a.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"exploratory": True, "arms": arms, "results": results,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                                text=True).stdout.strip()}
        if summary:
            payload["summary"] = summary
        a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with h5py.File(a.pooled_cache, "r") as h5:
        m = {k: h5[k][:] for k in ("y", "subject", "run_id", "char_idx", "seq_idx", "stim_code",
                                    "study", "lit_mask")}
        for k in ("subject", "run_id", "study"):
            m[k] = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in m[k]])
        channels, sfreq, tmin = str(h5.attrs["channels"]).split(","), float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        X, y = h5["X"], m["y"]
        subjects = {s: sorted(set(m["subject"][m["study"] == s])) for s in "DEG"}
        held = {s: v[:a.limit_subjects] if a.limit_subjects else v for s, v in subjects.items()}

        def fit(train_mask, pred_idx, seed):
            _, _, sc, kept, n_train = _fit_m0_streaming(X, np.flatnonzero(train_mask), pred_idx, y, cfg,
                                                        channels, sfreq, tmin, seed)
            full = np.full(len(y), np.nan)
            full[kept] = sc
            return full, int(n_train)

        # d_only for E/G: one shared fit over all D, scored on every held-out E/G subject
        if "d_only" in arms:
            eg = [s for st in "EG" for s in held[st]]
            if any(f"d_only|{s}" not in results for s in eg):
                pred = np.flatnonzero(np.isin(m["subject"], eg))
                sc, n_train = fit(m["study"] == "D", pred, 1 + a.seed_offset)   # shared d_only fit
                for st in "EG":
                    for s in held[st]:
                        idx = np.flatnonzero(m["subject"] == s)
                        results[f"d_only|{s}"] = {"study": st, "arm": "d_only", "train_epochs": n_train,
                                                  **evaluate(m, st, idx, sc)}
                save()
        for fold, (st, s) in enumerate([(st, s) for st in "DEG" for s in held[st]], 10):
            idx = np.flatnonzero(m["subject"] == s)
            for arm in arms:
                key = f"{arm}|{s}"
                if key in results or (arm == "d_only" and st != "D") or (arm == "own_study" and st == "D"):
                    continue
                if arm == "pooled":
                    train = m["subject"] != s
                elif arm == "d_only":
                    train = (m["study"] == "D") & (m["subject"] != s)
                elif arm == "own_study":
                    train = (m["study"] == st) & (m["subject"] != s)
                else:
                    raise ValueError(arm)
                sc, n_train = fit(train, idx, fold + a.seed_offset)             # per-fold fit
                results[key] = {"study": st, "arm": arm, "train_epochs": n_train, **evaluate(m, st, idx, sc)}
                print(f"{key}: acc={results[key]['full_accuracy']} auc={results[key]['auc']}", flush=True)
                save()

    summary, comps = {}, []
    for st in "DEG":
        for arm in arms:
            vals = [r["full_accuracy"] for r in results.values()
                    if r["study"] == st and r["arm"] == arm and r["full_accuracy"] is not None]
            if vals:
                summary[f"{st}|{arm}"] = {"subjects": len(vals), "mean_accuracy": float(np.mean(vals))}
        for x, z in (("pooled", "d_only"), ("pooled", "own_study")):
            if x in arms and z in arms and not (st == "D" and z == "own_study"):
                sx = {r["subject"]: r for r in []}
                a_v = [results[f"{x}|{s}"]["full_accuracy"] for s in held[st]
                       if f"{x}|{s}" in results and f"{z}|{s}" in results]
                b_v = [results[f"{z}|{s}"]["full_accuracy"] for s in held[st]
                       if f"{x}|{s}" in results and f"{z}|{s}" in results]
                if len(a_v) >= 2:
                    comps.append({"study": st, "comparison": f"{x} - {z}", **compare(a_v, b_v)})
    for c, adj in zip(comps, holm([c["wilcoxon_p"] for c in comps]) if comps else []):
        c["holm_p"] = adj
    summary["comparisons"] = comps
    save(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()