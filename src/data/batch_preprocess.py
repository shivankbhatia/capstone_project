"""
Day 2 — Preprocessing
Bandpass/notch filtering, epoching (-100 to 800 ms around stimulus onset),
baseline correction. Parses bigP3BCI EDF files directly into epoched EEG
data + ground-truth spelled text for LLM context.

IMPORTANT: sequences_per_selection=20 is only valid for FIXED-sequence
conditions (RC/Train calibration, confirmed empirically). Study D's
Dyn/DynBigram conditions use ADAPTIVE stopping -- CurrentTarget changes
value MID-FILE (confirmed via low whole-file vote agreement -- 18-39%
across every adaptive file inspected, with vote distributions showing
multiple comparable-sized clusters rather than one dominant code with
noise). Each Dyn/DynBigram file contains MULTIPLE character selections,
not one -- segment by CurrentTarget transitions (see
segment_by_target_transitions) and vote WITHIN each run, not across the
whole file.
"""

import json
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mne

import warnings
warnings.filterwarnings(
    "ignore",
    message="Channels contain different (highpass|lowpass) filters.*",
    category=RuntimeWarning,
)

from tqdm import tqdm

mne.set_log_level('ERROR')

LOW_ADAPTIVE_AGREEMENT_THRESHOLD = 0.50


def segment_by_target_transitions(target_codes_raw, min_run_length=1):
    """
    Splits a Dyn/DynBigram file's per-flash CurrentTarget codes into
    contiguous runs, each run corresponding to one character selection
    (confirmed: whole-file majority voting produced 18-39% agreement with
    multiple comparable-sized vote clusters -- consistent with several
    distinct characters being spelled per file, not one).

    Returns a list of (code, run_length, agreement_frac) tuples, one per
    detected character, using majority vote WITHIN each run rather than
    across the whole file.
    """
    runs = []
    current_run = [int(target_codes_raw[0])]
    for code in target_codes_raw[1:]:
        code = int(code)
        if code == current_run[-1]:
            current_run.append(code)
        else:
            runs.append(current_run)
            current_run = [code]
    runs.append(current_run)

    results = []
    for run in runs:
        if len(run) < min_run_length:
            continue
        code, agreement_count = Counter(run).most_common(1)[0]
        results.append((code, len(run), agreement_count / len(run)))
    return results


def parse_bigp3bci_edf(edf_path, tmin=-0.1, tmax=0.8, l_freq=0.1, h_freq=30.0,
                           notch_freq=None, sequences_per_selection=20, verbose=False):
    """
    Parses bigP3BCI EDF files into epoched EEG data.

    notch_freq: line-noise frequency to notch out (50 or 60 Hz). Left as
    None by default -- confirm which applies to this dataset's recording
    site before setting it, rather than guessing.

    sequences_per_selection: number of consecutive target flashes that
    belong to one character selection. Confirmed at 20 for FIXED-sequence
    conditions (RC/Train) only -- Dyn/DynBigram (adaptive stopping) files
    use segment_by_target_transitions instead, see module docstring.
    """
    if verbose:
        print(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    all_channels_raw = raw.ch_names

    is_adaptive = "Dyn" in str(edf_path)  # covers both Dyn and DynBigram
    if is_adaptive:
        print("NOTE: filename contains 'Dyn' -- adaptive-stopping condition. "
              "Segmenting by CurrentTarget transitions (multiple characters "
              "per file, variable sequence count per character), NOT a "
              "single whole-file majority vote.")

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
    if verbose:
        print(f"Applying bandpass filter ({l_freq}-{h_freq} Hz) to {len(actual_eeg_channels)} EEG channels...")
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks=actual_eeg_channels, verbose=False)

    if notch_freq is not None:
        if verbose:
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
    if verbose:
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

    current_target_trace = raw.get_data(picks=[current_target_ch])[0]
    target_indices = events[events[:, 2] == 1, 0]
    target_codes_raw = np.round(current_target_trace[target_indices]).astype(int)

    n_targets = len(target_codes_raw)
    agreement_info = {
        "is_adaptive": is_adaptive,
        "target_flash_count": int(n_targets),
    }

    if is_adaptive:
        # Adaptive-stopping files: CurrentTarget changes value mid-file --
        # each contiguous run is one character selection, decoded over a
        # VARIABLE number of sequences (that's the entire point of adaptive
        # stopping). Vote WITHIN each run, not across the whole file. See
        # module docstring for how this was confirmed.
        char_runs = segment_by_target_transitions(target_codes_raw)
        sequence_codes = [code for code, _, _ in char_runs]

        per_char_agreement = [agreement for _, _, agreement in char_runs]
        min_agreement = min(per_char_agreement) if per_char_agreement else 0.0
        agreement_info.update({
            "n_characters_detected": len(char_runs),
            "run_lengths": [run_len for _, run_len, _ in char_runs],
            "per_char_agreement": per_char_agreement,
            "min_char_agreement": float(min_agreement),
            "below_threshold": bool(min_agreement < LOW_ADAPTIVE_AGREEMENT_THRESHOLD),
        })

        print(f"  Detected {len(char_runs)} character(s) via transition segmentation "
              f"over {n_targets} target flashes")
        for i, (code, run_len, agreement) in enumerate(char_runs):
            if agreement < LOW_ADAPTIVE_AGREEMENT_THRESHOLD:
                print(f"    Char {i}: code={code}, run_length={run_len}, "
                      f"LOW agreement ({agreement:.1%})")
    else:
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
    if verbose:
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

    return epochs, spelled_string, {"grid_map": grid_map, "n_rows": n_rows, "n_cols": n_cols, "agreement": agreement_info}


def _study_d_quality_context(edf_path):
    path = Path(edf_path)
    parts = path.parts
    study = next((part for part in parts if part in {"StudyD", "StudyE"}), "Unknown")
    study_index = parts.index(study) if study in parts else -1

    subject = parts[study_index + 1] if study_index >= 0 and study_index + 1 < len(parts) else "Unknown"
    session = parts[study_index + 2] if study_index >= 0 and study_index + 2 < len(parts) else "Unknown"
    phase = next((part for part in parts if part in {"Train", "Test"}), "Unknown")

    condition = path.parent.name
    if condition in {"Train", "Test"}:
        condition = "Unspecified"

    return {
        "study": study,
        "subject": subject,
        "session": session,
        "phase": phase,
        "condition": condition,
    }


def _quality_flag_row(edf_path, flag_type, detail):
    path = Path(edf_path)
    return {
        **_study_d_quality_context(path),
        "file": str(path),
        "file_name": path.name,
        "flag_type": flag_type,
        "detail": detail,
    }


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _append_failure_log(log_path, edf_path, exc):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {edf_path}\n")
        handle.write(f"{type(exc).__name__}: {exc}\n")
        handle.write(traceback.format_exc())
        handle.write("\n")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    raw_root = project_root / "data" / "raw" / "bigP3BCI_dataset" / "bigP3BCI-data" / "StudyD"
    processed_root = project_root / "data" / "processed"
    study_processed_root = processed_root / "StudyD"
    grid_layout_path = processed_root / "grid_layout.json"
    registry_path = processed_root / "ground_truth_registry.json"
    quality_flags_path = processed_root / "quality_flags.csv"
    failure_log_path = processed_root / "batch_preprocess_failures.log"

    edf_files = sorted(raw_root.rglob("*.[eE][dD][fF]"))
    if not edf_files:
        raise FileNotFoundError(f"No Study D EDF files found under {raw_root}")

    processed_count = 0
    failed_count = 0
    adaptive_file_count = 0
    low_agreement_count = 0

    grid_layouts = json.loads(grid_layout_path.read_text()) if grid_layout_path.exists() else {}
    ground_truth_registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    quality_flags = []

    print(f"Found {len(edf_files)} Study D EDF files under {raw_root}")
    for edf_path in tqdm(edf_files, desc="Preprocessing", unit="file"):
        output_path = study_processed_root / f"{edf_path.stem}-epo.fif"

        if output_path.exists():
            continue

        try:
            epochs, ground_truth_text, grid_info = parse_bigp3bci_edf(str(edf_path), verbose=False)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            epochs.save(str(output_path), overwrite=True)

            static_info = {"grid_map": grid_info["grid_map"], "n_rows": grid_info["n_rows"], "n_cols": grid_info["n_cols"]}
            if "StudyD" not in grid_layouts:
                grid_layouts["StudyD"] = static_info
            elif grid_layouts["StudyD"] != static_info:
                quality_flags.append(_quality_flag_row(
                    edf_path,
                    "grid_layout_mismatch",
                    "Grid layout differs from the first observed Study D layout; "
                    "continuing preprocessing and retaining the original canonical layout in grid_layout.json.",
                ))

            ground_truth_registry[edf_path.stem] = ground_truth_text

            agreement = grid_info.get("agreement", {})
            if agreement.get("is_adaptive"):
                adaptive_file_count += 1
                n_chars = int(agreement.get("n_characters_detected", 0))
                min_agreement = float(agreement.get("min_char_agreement", 0.0))
                quality_flags.append(_quality_flag_row(
                    edf_path,
                    "adaptive_multi_character_segmentation",
                    f"Detected {n_chars} character(s) via CurrentTarget transitions; "
                    f"min per-character agreement {min_agreement:.1%}.",
                ))
                if agreement.get("below_threshold"):
                    low_agreement_count += 1
                    quality_flags.append(_quality_flag_row(
                        edf_path,
                        "low_currenttarget_agreement",
                        f"Min per-character agreement {min_agreement:.1%} is below "
                        f"{LOW_ADAPTIVE_AGREEMENT_THRESHOLD:.0%} for at least one character.",
                    ))

            _write_json(grid_layout_path, grid_layouts)
            _write_json(registry_path, ground_truth_registry)

            processed_count += 1
            tqdm.write(f"Saved epochs to {output_path}")
        except Exception as exc:
            failed_count += 1
            _append_failure_log(failure_log_path, edf_path, exc)
            tqdm.write(f"FAILED: {edf_path.name} ({type(exc).__name__}: {exc})")

    if quality_flags:
        pd.DataFrame(quality_flags).to_csv(quality_flags_path, index=False)

    print("\n--- Batch Preprocess Summary ---")
    print(f"Processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Grid layout: {grid_layout_path}")
    print(f"Ground-truth registry: {registry_path}")
    if adaptive_file_count:
        print(
            f"Adaptive files with a low-agreement character: "
            f"{low_agreement_count}/{adaptive_file_count} "
            f"({(low_agreement_count / adaptive_file_count):.1%})"
        )
        print(f"Quality flags: {quality_flags_path}")
    if failed_count:
        print(f"Failure log: {failure_log_path}")