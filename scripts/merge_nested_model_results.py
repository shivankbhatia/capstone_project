#!/usr/bin/env python3
"""Merge independently generated nested model score sidecars."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--add-json", type=Path, required=True)
    parser.add_argument("--add-scores", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_json.read_text())
    addition = json.loads(args.add_json.read_text())
    base.update(addition)
    args.base_json.write_text(json.dumps(base, indent=2) + "\n")
    with np.load(args.base_scores, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(args.add_scores, allow_pickle=False) as data:
        arrays.update({key: data[key] for key in data.files})
    np.savez_compressed(args.base_scores, **arrays)


if __name__ == "__main__":
    main()
