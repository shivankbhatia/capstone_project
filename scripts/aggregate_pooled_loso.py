#!/usr/bin/env python3
"""Average per-subject accuracy over seed runs of run_pooled_loso.py and compare arms.

Primary family (declared here, before the multi-seed results are seen): pooled - d_only on D, E, G (Holm over 3).
Secondary family: pooled - own_study on E, G (Holm over 2). Exploratory.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_pooled_loso import compare, holm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    runs = [json.loads(p.read_text())["results"] for p in a.inputs]
    keys = set.intersection(*(set(r) for r in runs))
    per = {}
    for k in keys:
        arm, subject = k.split("|")
        vals = [r[k]["full_accuracy"] for r in runs]
        per[(arm, subject)] = {"study": runs[0][k]["study"], "mean": float(np.mean(vals)),
                               "sd": float(np.std(vals)), "values": vals}

    def subjects(study, *arms):
        return sorted(s for (arm, s), v in per.items()
                      if arm == arms[0] and v["study"] == study and all((x, s) in per for x in arms))

    out = {"seeds": len(runs), "inputs": [str(p) for p in a.inputs], "means": {}, "primary": [], "secondary": []}
    for study in "DEG":
        for arm in ("pooled", "d_only", "own_study"):
            vs = [per[(arm, s)]["mean"] for s in subjects(study, arm)]
            if vs:
                per_seed = [float(np.mean([per[(arm, s)]["values"][i] for s in subjects(study, arm)]))
                            for i in range(len(runs))]
                out["means"][f"{study}|{arm}"] = {"subjects": len(vs), "mean": float(np.mean(vs)),
                                                  "mean_per_seed": per_seed}

    def family(pairs, name):
        rows = []
        for study, x, z in pairs:
            ss = subjects(study, x, z)
            if len(ss) >= 2:
                rows.append({"study": study, "comparison": f"{x} - {z}",
                             **compare([per[(x, s)]["mean"] for s in ss], [per[(z, s)]["mean"] for s in ss])})
        for r, p in zip(rows, holm([r["wilcoxon_p"] for r in rows]) if rows else []):
            r["holm_p"] = p
        out[name] = rows

    family([("D", "pooled", "d_only"), ("E", "pooled", "d_only"), ("G", "pooled", "d_only")], "primary")
    family([("E", "pooled", "own_study"), ("G", "pooled", "own_study")], "secondary")
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("means", "primary", "secondary")}, indent=2))


if __name__ == "__main__":
    main()