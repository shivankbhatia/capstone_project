"""
Day 2 — Preprocessing
Bandpass/notch filtering, epoching (-100 to 800 ms around stimulus onset),
baseline correction. Parses bigP3BCI EDF files directly into epoched EEG
data + ground-truth spelled text for LLM context.

IMPORTANT: sequences_per_selection=20 is only valid for FIXED-sequence
conditions (RC/Train calibration, confirmed empirically). Study D's
Dyn/DynBigram conditions use ADAPTIVE stopping -- a variable number of
sequences per character -- so this block-based segmentation will silently
produce wrong ground truth on those files. Don't run this on Dyn/DynBigram
paths until we've built proper variable-length segmentation for them.
"""

import numpy as np
import pandas as pd
import mne

def parse_bigp3bci_edf(edf_path, tmin=-0.1, tmax=0.8, l_freq=0.1, h_freq=30.0,
                           notch_freq=None, sequences_per_selection=20):
    """
    Parses bigP3BCI EDF files into epoched EEG data.

    notch_freq: line-noise frequency to notch out (50 or 60 Hz). Left as
    None by default -- confirm which applies to this dataset's recording
    site before setting it, rather than guessing.

    sequences_per_selection: number of consecutive target flashes that
    belong to one character selection. Confirmed at 20 for FIXED-sequence
    conditions (RC/Train) only -- do NOT use this for Dyn/DynBigram
    (adaptive stopping) files, see module docstring.
    """
    print(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    all_channels_raw = raw.ch_names

    if "Dyn" in str(edf_path):
        print("WARNING: filename contains 'Dyn' -- this looks like an "
              "adaptive-stopping condition. Fixed-block segmentation "
              "(sequences_per_selection) is known to be WRONG for these "
              "files. Ground truth extracted below should be treated as "
              "unreliable until variable-length segmentation is implemented.")

    # -------------------------------------------------------------------------
    # 1. CHANNEL RENAMING & CLASSIFICATION
    # -------------------------------------------------------------------------
    actual_eeg_channels = []
    char_channels = []
    grid_map = {}
    rename_mapping = {}

    for ch in all_channels_raw:
        clean_ch = ch.strip()

        if clean_ch.startswith('EEG_'):
            std_name = clean_ch.replace('EEG_', '')
            rename_mapping[ch] = std_name
            actual_eeg_channels.append(std_name)
            continue

        # Split from the right rather than using a character-class regex --
        # some grids (e.g. Study D's extended keyboard layout) have
        # punctuation-only labels like "'_8_2", "?_6_1", "\_6_4" that a
        # [a-zA-Z0-9_]+ pattern would silently fail to match and drop.
        parts = clean_ch.rsplit('_', 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            label, row, col = parts[0], int(parts[1]), int(parts[2])
            grid_map[label] = (row, col)
            char_channels.append(ch)

    if not rename_mapping:
        raise ValueError(f"Could not find any 'EEG_' prefixed channels. Raw channels: {all_channels_raw}")

    raw.rename_channels(rename_mapping)

    stim_begin_ch = next((ch for ch in all_channels_raw if ch.strip().lower() == 'stimulusbegin'), None)
    stim_type_ch = next((ch for ch in all_channels_raw if ch.strip().lower() == 'stimulustype'), None)
    stim_code_ch = next((ch for ch in all_channels_raw if ch.strip().lower() == 'stimuluscode'), None)
    current_target_ch = next((ch for ch in all_channels_raw if ch.strip().lower() == 'currenttarget'), None)

    if not all([stim_begin_ch, stim_type_ch, stim_code_ch, current_target_ch]):
        raise ValueError("Missing essential trigger channels in the EDF file.")

    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    # -------------------------------------------------------------------------
    # 2. FILTERING
    # -------------------------------------------------------------------------
    print(f"Applying bandpass filter ({l_freq}-{h_freq} Hz) to {len(actual_eeg_channels)} EEG channels...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=actual_eeg_channels, verbose=False)

    if notch_freq is not None:
        print(f"Applying notch filter at {notch_freq} Hz...")
        raw.notch_filter(freqs=notch_freq, picks=actual_eeg_channels, verbose=False)

    # -------------------------------------------------------------------------
    # 3. EVENT EXTRACTION
    # -------------------------------------------------------------------------
    stim_begin = raw.get_data(picks=[stim_begin_ch])[0]
    stim_type = raw.get_data(picks=[stim_type_ch])[0]
    stim_code = raw.get_data(picks=[stim_code_ch])[0]

    threshold = 0.5
    rising_edges = np.where((stim_begin[:-1] <= threshold) & (stim_begin[1:] > threshold))[0] + 1

    events = []
    stimulus_codes_per_event = []
    for sample_idx in rising_edges:
        is_target = int(stim_type[sample_idx] > 0.5)
        events.append([sample_idx, 0, is_target])
        stimulus_codes_per_event.append(int(np.round(stim_code[sample_idx])))

    events = np.array(events, dtype=int)
    stimulus_codes_per_event = np.array(stimulus_codes_per_event, dtype=int)
    print(f"Extracted {len(events)} flash events ({np.sum(events[:, 2] == 1)} Targets, {np.sum(events[:, 2] == 0)} Non-Targets).")

    # -------------------------------------------------------------------------
    # 4. LLM CONTEXT EXTRACTION
    # -------------------------------------------------------------------------
    # Real index -> character mapping from grid_map (channel-derived), not a
    # hardcoded alphabet -- this also correctly handles non-alphabetic labels
    # like 'Sp' (space) at whatever grid position they actually sit.
    # Encoding is idx = (row - 1) * n_cols + col (col is NOT shifted by -1),
    # confirmed against user-verified ground truth: T_4_2->20, H_2_2->8,
    # E_1_5->5 matches the spelled word "THE". Index 0 is unreachable under
    # this formula (min is row=1,col=1 -> 1), so a CurrentTarget of 0 means
    # "no active target" (idle/reset), not a real character.
    n_rows = max(row for row, col in grid_map.values())
    n_cols = max(col for row, col in grid_map.values())
    idx_to_char = {
        (row - 1) * n_cols + col: label
        for label, (row, col) in grid_map.items()
    }

    # Each file (in FIXED-sequence conditions) contains MULTIPLE characters,
    # each selected via a fixed number of consecutive sequences. Segment by
    # position in chronological order, NOT by aggregating votes across the
    # whole file (order matters -- e.g. "THE" vs sorting codes numerically)
    # and NOT by "did the code change" (which would merge a genuine repeated
    # letter). SEE MODULE DOCSTRING: invalid for Dyn/DynBigram files.
    current_target_trace = raw.get_data(picks=[current_target_ch])[0]
    target_indices = events[events[:, 2] == 1, 0]
    target_codes_raw = np.round(current_target_trace[target_indices]).astype(int)

    from collections import Counter
    n_targets = len(target_codes_raw)
    if n_targets % sequences_per_selection != 0:
        print(f"WARNING: {n_targets} target flashes doesn't divide evenly by "
              f"sequences_per_selection={sequences_per_selection} -- block "
              f"boundaries below may be off for the last block.")

    n_blocks = n_targets // sequences_per_selection
    sequence_codes = []
    for i in range(n_blocks):
        block = target_codes_raw[i * sequences_per_selection: (i + 1) * sequences_per_selection]
        code, agreement_count = Counter(block.tolist()).most_common(1)[0]
        agreement_frac = agreement_count / len(block)
        if agreement_frac < 0.9:
            print(f"  Block {i}: LOW agreement ({agreement_frac:.1%}) -- votes: {dict(Counter(block.tolist()))}")
        sequence_codes.append(code)

    spelled_string = "".join(
        idx_to_char.get(code, f"[{code}]") for code in sequence_codes
    )

    print(f"Extracted Ground-Truth Spelled String: '{spelled_string}'")

    # -------------------------------------------------------------------------
    # 5. EPOCHING
    # -------------------------------------------------------------------------
    event_id = {'non_target': 0, 'target': 1}

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        picks=actual_eeg_channels,
        baseline=(tmin, 0),
        preload=True,
        verbose=False
    )

    # CRITICAL: preserve StimulusCode (which row/col physically flashed) as
    # metadata. epochs.events[:, 2] only holds the binary target/non-target
    # label used for event_id above -- without this, downstream code (e.g.
    # run_pipeline.py's grid reconstruction) has no way to know which grid
    # cell each epoch corresponds to. mne.Epochs can silently drop epochs
    # too close to the recording edges, so index back via epochs.selection
    # rather than assuming a 1:1 match with the original events array.
    metadata = pd.DataFrame({
        "stimulus_code": stimulus_codes_per_event,
        "stimulus_type": events[:, 2],
    }).iloc[epochs.selection].reset_index(drop=True)
    epochs.metadata = metadata

    return epochs, spelled_string, {"grid_map": grid_map, "n_rows": n_rows, "n_cols": n_cols}


if __name__ == "__main__":
    sample_edf = "./data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyD/D_09/SE001/Test/Dyn/D_09_SE001_Dyn_Test04.edf"

    epochs, ground_truth_text, grid_info = parse_bigp3bci_edf(sample_edf)

    print("\n--- Parser Output Summary ---")
    print(f"Epochs shape: {epochs.get_data().shape} (Trials x Channels x Timepoints)")
    print(f"EEG Channels: {epochs.ch_names}")
    print(f"Target Ratio: {len(epochs['target'])} / {len(epochs)}")
    print(f"Text for LLM Context: '{ground_truth_text}'")
    print(f"Grid: {grid_info['n_rows']} rows x {grid_info['n_cols']} cols")