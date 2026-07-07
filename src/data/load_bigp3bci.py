"""
bigP3BCI (PhysioNet) — dataset inspection script.

Step 1: figure out exactly what's in the downloaded subject folders before
writing the real parser. bigP3BCI packs event info (target/non-target,
row/column flashed, current target character, etc.) into BCI2000-style
auxiliary channels inside the EDF+ files rather than standard annotations,
and the exact channel set/grid size varies by study. So: point this at
your downloaded folder, run it, and share the printed output before we
write src/data/parse_bigp3bci.py against real assumptions instead of
guessed ones.

Reference (for related-work positioning, not just data):
Mainsah et al., language-model-assisted dynamic stopping in P300 spellers.

Usage:
    python -m src.data.load_bigp3bci /path/to/your/downloaded/folder
"""

import sys
from pathlib import Path

import mne

# Known BCI2000 P3SpellerTask state-variable names that commonly show up
# as auxiliary "channels" inside these EDF+ files. Not exhaustive — the
# printout below will show us the real names in your files either way.
LIKELY_EVENT_CHANNEL_NAMES = {
    "stimuluscode",
    "stimulustype",
    "stimulusbegin",
    "phaseinsequence",
    "currenttarget",
    "selectedtarget",
    "targetcode",
    "flashing",
    "fakefeedback",
}


def find_edf_files(root: Path):
    return sorted(root.rglob("*.edf"))


def inspect_file(path: Path):
    print(f"\n{'=' * 70}")
    print(f"File: {path}")
    print(f"{'=' * 70}")

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")

    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.n_times / raw.info['sfreq']:.1f} s")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"All channel names: {raw.ch_names}")

    likely_event_channels = [
        ch for ch in raw.ch_names if ch.strip().lower() in LIKELY_EVENT_CHANNEL_NAMES
    ]
    print(f"\nLikely event/state channels found: {likely_event_channels}")

    if len(raw.annotations) > 0:
        print(f"\nEmbedded annotations found: {len(raw.annotations)}")
        print(f"Unique annotation descriptions: {set(raw.annotations.description)}")
    else:
        print("\nNo embedded standard annotations (expected for this dataset — "
              "event info likely lives in the auxiliary channels above instead).")

    if likely_event_channels:
        raw.load_data(verbose="ERROR")
        for ch in likely_event_channels:
            data = raw.get_data(picks=[ch])[0]
            unique_vals = sorted(set(data.tolist()))
            preview = unique_vals[:10]
            print(f"  '{ch}' unique values (first 10 shown): {preview}"
                  f"{' ...' if len(unique_vals) > 10 else ''}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.data.load_bigp3bci /path/to/downloaded/folder")
        sys.exit(1)

    root = Path(sys.argv[1]).expanduser().resolve()
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    edf_files = find_edf_files(root)
    print(f"Found {len(edf_files)} .edf files under {root}")

    if not edf_files:
        print("No .edf files found — double check the path, or that the "
              "download extracted correctly (should contain Train/Test "
              "subfolders per subject/session).")
        sys.exit(1)

    seen_dirs = set()
    for f in edf_files:
        if f.parent not in seen_dirs:
            inspect_file(f)
            seen_dirs.add(f.parent)

    print(f"\n{'=' * 70}")
    print(f"Total .edf files: {len(edf_files)} across {len(seen_dirs)} folders")
    print("Share this output and we'll write the real parser next.")


if __name__ == "__main__":
    main()