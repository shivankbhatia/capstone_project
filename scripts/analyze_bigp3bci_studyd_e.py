#!/usr/bin/env python3
"""
Generate reusable StudyD/StudyE analytics for the bigP3BCI dataset.

The script scans EDF files directly and reads only the auxiliary/state
channels needed for event analytics. It writes:
  - bigp3bci_studyd_e_findings.md
  - file_summary.csv
  - character_trials.csv
  - stimulus_code_counts.csv
  - grid_layouts.csv
  - quality_flags.csv

Example:
  python scripts/analyze_bigp3bci_studyd_e.py \
    --data-root data/raw/bigP3BCI_dataset \
    --output-dir results/bigp3bci_studyd_e_findings
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

MPL_CACHE_DIR = Path("results/.matplotlib-cache").resolve()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import mne
import numpy as np
import pandas as pd


EVENT_CHANNELS = {
    "stimulusbegin": "stimulus_begin",
    "stimulustype": "stimulus_type",
    "stimuluscode": "stimulus_code",
    "currenttarget": "current_target",
}

DEFAULT_ROOT = Path("data/raw/bigP3BCI_dataset")
DEFAULT_OUTPUT = Path("results/bigp3bci_studyd_e_findings")
FALLBACK_TARGET_FLASHES_PER_SELECTION = 20
LOW_AGREEMENT_THRESHOLD = 0.90


@dataclass(frozen=True)
class FileContext:
    study: str
    subject: str
    session: str
    phase: str
    condition: str


def normalize_channel_name(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def find_channel(raw: mne.io.BaseRaw, canonical_name: str) -> Optional[str]:
    for channel in raw.ch_names:
        if normalize_channel_name(channel) == canonical_name:
            return channel
    return None


def find_edf_files(data_root: Path, studies: Sequence[str]) -> List[Path]:
    study_names = set(studies)
    return sorted(
        path for path in data_root.rglob("*.edf")
        if any(part in study_names for part in path.parts)
    )


def parse_context(path: Path) -> FileContext:
    parts = path.parts
    study = next((part for part in parts if part in {"StudyD", "StudyE"}), "Unknown")
    study_index = parts.index(study) if study in parts else -1

    subject = parts[study_index + 1] if study_index >= 0 and study_index + 1 < len(parts) else "Unknown"
    session = parts[study_index + 2] if study_index >= 0 and study_index + 2 < len(parts) else "Unknown"
    phase = next((part for part in parts if part in {"Train", "Test"}), "Unknown")

    condition = path.parent.name
    if condition in {"Train", "Test"}:
        condition = "Unspecified"

    return FileContext(study=study, subject=subject, session=session, phase=phase, condition=condition)


def parse_grid_channels(channel_names: Iterable[str]) -> Tuple[Dict[str, Tuple[int, int]], int, int]:
    grid_map: Dict[str, Tuple[int, int]] = {}
    for channel in channel_names:
        clean = channel.strip()
        parts = clean.rsplit("_", 2)
        if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        label, row, col = parts[0], int(parts[1]), int(parts[2])
        if label and not normalize_channel_name(label).startswith("eeg"):
            grid_map[label] = (row, col)

    if not grid_map:
        return {}, 0, 0

    n_rows = max(row for row, _ in grid_map.values())
    n_cols = max(col for _, col in grid_map.values())
    return grid_map, n_rows, n_cols


def grid_code_to_label(code: int, grid_map: Dict[str, Tuple[int, int]], n_cols: int) -> str:
    if code <= 0 or n_cols <= 0:
        return f"[{code}]"

    for label, (row, col) in grid_map.items():
        if (row - 1) * n_cols + col == code:
            return label
    return f"[{code}]"


def rising_edges(signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if signal.size < 2:
        return np.array([], dtype=int)
    return np.where((signal[:-1] <= threshold) & (signal[1:] > threshold))[0] + 1


def safe_int_values(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=int)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.rint(values).astype(int)


def most_common_with_agreement(values: Sequence[int]) -> Tuple[int, int, float]:
    if not values:
        return 0, 0, 0.0
    code, count = Counter(values).most_common(1)[0]
    return int(code), int(count), float(count / len(values))


def infer_character_trials(
    path: Path,
    context: FileContext,
    target_codes: np.ndarray,
    grid_map: Dict[str, Tuple[int, int]],
    n_cols: int,
    fallback_target_flashes_per_selection: int,
) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    flags: List[dict] = []
    is_adaptive = "Dyn" in str(path)

    if target_codes.size == 0:
        flags.append(flag_row(path, context, "no_target_flashes", "No target flashes found."))
        return rows, flags

    blocks: List[np.ndarray]
    segmentation_note: str
    if is_adaptive:
        blocks = [target_codes]
        segmentation_note = "adaptive_currenttarget_distribution"
    else:
        block_size = target_flashes_per_selection(context, fallback_target_flashes_per_selection)
        usable = (target_codes.size // block_size) * block_size
        if usable != target_codes.size:
            flags.append(flag_row(
                path,
                context,
                "fixed_block_remainder",
                f"{target_codes.size} target flashes is not divisible by "
                f"{block_size}. Last "
                f"{target_codes.size - usable} target flashes are not assigned to a character block.",
            ))
        blocks = [
            target_codes[start:start + block_size]
            for start in range(0, usable, block_size)
        ]
        segmentation_note = f"fixed_blocks_{block_size}_target_flashes"

    for index, block in enumerate(blocks, start=1):
        code, count, agreement = most_common_with_agreement(block.tolist())
        if agreement < LOW_AGREEMENT_THRESHOLD:
            flag_type = "adaptive_currenttarget_varies" if is_adaptive else "low_currenttarget_agreement"
            flags.append(flag_row(
                path,
                context,
                flag_type,
                f"Character block {index} majority agreement is {agreement:.1%}.",
            ))

        rows.append({
            **context_fields(context),
            "file": str(path),
            "file_name": path.name,
            "character_index": index,
            "inferred_target_code": code,
            "inferred_target_label": grid_code_to_label(code, grid_map, n_cols),
            "target_flash_count": int(block.size),
            "majority_count": count,
            "majority_agreement": agreement,
            "segmentation": segmentation_note,
        })

    return rows, flags


def target_flashes_per_selection(context: FileContext, fallback: int) -> int:
    if context.study == "StudyD" and context.condition == "RC":
        return 20
    if context.study == "StudyE" and context.condition == "CB":
        return 10
    return fallback


def context_fields(context: FileContext) -> dict:
    return {
        "study": context.study,
        "subject": context.subject,
        "session": context.session,
        "phase": context.phase,
        "condition": context.condition,
    }


def flag_row(path: Path, context: FileContext, flag_type: str, detail: str) -> dict:
    return {
        **context_fields(context),
        "file": str(path),
        "file_name": path.name,
        "flag_type": flag_type,
        "detail": detail,
    }


def analyze_file(path: Path, data_root: Path, fallback_target_flashes_per_selection: int) -> Tuple[dict, List[dict], List[dict], List[dict], List[dict]]:
    context = parse_context(path)
    flags: List[dict] = []
    character_rows: List[dict] = []
    stimulus_rows: List[dict] = []
    grid_rows: List[dict] = []

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    duration_sec = float(raw.n_times / sfreq) if sfreq else 0.0
    channel_lookup = {alias: find_channel(raw, canonical) for canonical, alias in EVENT_CHANNELS.items()}
    missing = [alias for alias, channel in channel_lookup.items() if channel is None]

    grid_map, n_rows, n_cols = parse_grid_channels(raw.ch_names)
    eeg_channel_count = sum(1 for ch in raw.ch_names if ch.strip().startswith("EEG_"))
    aux_channel_count = len(raw.ch_names) - eeg_channel_count

    if missing:
        flags.append(flag_row(path, context, "missing_event_channels", ", ".join(missing)))

    if not grid_map:
        flags.append(flag_row(path, context, "missing_grid_channels", "No character grid channels were detected."))

    expected_cells = n_rows * n_cols
    missing_grid_cells = expected_cells - len(grid_map) if expected_cells else 0
    if missing_grid_cells:
        flags.append(flag_row(path, context, "incomplete_grid", f"{missing_grid_cells} grid cells have no detected label."))

    for label, (row, col) in sorted(grid_map.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
        grid_rows.append({
            **context_fields(context),
            "file": str(path),
            "label": label,
            "row": row,
            "col": col,
            "target_code": (row - 1) * n_cols + col if n_cols else 0,
        })

    event_count = 0
    target_count = 0
    non_target_count = 0
    unique_stimulus_codes = 0
    min_stimulus_code = math.nan
    max_stimulus_code = math.nan
    target_flash_currenttarget_codes: np.ndarray = np.array([], dtype=int)

    if not missing:
        picks = [
            channel_lookup["stimulus_begin"],
            channel_lookup["stimulus_type"],
            channel_lookup["stimulus_code"],
            channel_lookup["current_target"],
        ]
        data = raw.get_data(picks=picks)
        stim_begin, stim_type, stim_code, current_target = data
        edges = rising_edges(stim_begin)
        event_count = int(edges.size)
        stim_type_events = safe_int_values(stim_type[edges])
        stim_code_events = safe_int_values(stim_code[edges])
        current_target_events = safe_int_values(current_target[edges])
        target_mask = stim_type_events > 0

        target_count = int(np.sum(target_mask))
        non_target_count = int(event_count - target_count)
        target_flash_currenttarget_codes = current_target_events[target_mask]

        if event_count == 0:
            flags.append(flag_row(path, context, "no_flash_events", "StimulusBegin had no rising edges."))

        if stim_code_events.size:
            unique_codes = sorted(set(stim_code_events.tolist()))
            unique_stimulus_codes = len(unique_codes)
            min_stimulus_code = int(min(unique_codes))
            max_stimulus_code = int(max(unique_codes))
            for code, count in Counter(stim_code_events.tolist()).most_common():
                stimulus_rows.append({
                    **context_fields(context),
                    "file": str(path),
                    "file_name": path.name,
                    "stimulus_code": int(code),
                    "count": int(count),
                    "fraction": float(count / event_count) if event_count else 0.0,
                })

        expected_rc_codes = n_rows + n_cols if n_rows and n_cols else 0
        uses_row_col_codes = context.condition in {"RC", "Dyn", "DynBigram"}
        if uses_row_col_codes and expected_rc_codes and unique_stimulus_codes and unique_stimulus_codes != expected_rc_codes:
            flags.append(flag_row(
                path,
                context,
                "unexpected_stimulus_code_count",
                f"Found {unique_stimulus_codes} unique stimulus codes; row/column layout implies {expected_rc_codes}.",
            ))

        trial_rows, trial_flags = infer_character_trials(
            path,
            context,
            target_flash_currenttarget_codes,
            grid_map,
            n_cols,
            fallback_target_flashes_per_selection,
        )
        character_rows.extend(trial_rows)
        flags.extend(trial_flags)

    relative_path = str(path.relative_to(data_root)) if path.is_relative_to(data_root) else str(path)
    summary = {
        **context_fields(context),
        "file": str(path),
        "relative_path": relative_path,
        "file_name": path.name,
        "sfreq_hz": sfreq,
        "duration_sec": duration_sec,
        "channel_count": len(raw.ch_names),
        "eeg_channel_count": eeg_channel_count,
        "aux_channel_count": aux_channel_count,
        "grid_rows": n_rows,
        "grid_cols": n_cols,
        "grid_cell_count": len(grid_map),
        "missing_grid_cells": missing_grid_cells,
        "event_count": event_count,
        "target_flash_count": target_count,
        "non_target_flash_count": non_target_count,
        "target_fraction": float(target_count / event_count) if event_count else 0.0,
        "unique_stimulus_codes": unique_stimulus_codes,
        "min_stimulus_code": min_stimulus_code,
        "max_stimulus_code": max_stimulus_code,
        "inferred_character_count": len(character_rows),
        "is_adaptive_condition": "Dyn" in str(path),
        "missing_event_channels": ", ".join(missing),
    }
    return summary, character_rows, stimulus_rows, grid_rows, flags


def compact_stats(series: pd.Series) -> str:
    if series.empty:
        return "n/a"
    return (
        f"mean {series.mean():.1f}, median {series.median():.1f}, "
        f"min {series.min():.1f}, max {series.max():.1f}"
    )


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.loc[:, columns].head(max_rows).copy()
    shown = shown.fillna("")
    headers = [str(column) for column in shown.columns]
    rows = [[str(value) for value in row] for row in shown.to_numpy()]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(values: Sequence[str]) -> str:
        return "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])


def write_report(
    output_path: Path,
    data_root: Path,
    studies: Sequence[str],
    file_df: pd.DataFrame,
    char_df: pd.DataFrame,
    stim_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    flags_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# bigP3BCI StudyD/StudyE Findings")
    lines.append("")
    lines.append(f"Data root: `{data_root}`")
    lines.append(f"Studies requested: {', '.join(studies)}")
    lines.append("")

    if file_df.empty:
        lines.extend([
            "## No EDF Files Found",
            "",
            "No `.edf` files were found for StudyD or StudyE under the data root.",
            "Download or extract the bigP3BCI EDF files so the directory contains paths like "
            "`StudyD/<subject>/<session>/Train/.../*.edf` and `StudyE/<subject>/<session>/.../*.edf`, "
            "then rerun this script.",
            "",
            "Given this application is focused on StudyD and StudyE, no further studies are needed initially. "
            "Download additional studies only if you need broader cross-study generalization checks or more "
            "calibration data after StudyD/E coverage is confirmed.",
        ])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    total_hours = file_df["duration_sec"].sum() / 3600
    lines.extend([
        "## Executive Summary",
        "",
        f"- Files scanned: {len(file_df):,}",
        f"- Subjects: {file_df['subject'].nunique():,}",
        f"- Sessions: {file_df[['subject', 'session']].drop_duplicates().shape[0]:,}",
        f"- Recording duration: {total_hours:.2f} hours",
        f"- Flash events: {int(file_df['event_count'].sum()):,}",
        f"- Target flashes: {int(file_df['target_flash_count'].sum()):,}",
        f"- Inferred character trials: {int(file_df['inferred_character_count'].sum()):,}",
        f"- Quality flags: {len(flags_df):,}",
        "",
    ])

    by_study = file_df.groupby("study", dropna=False).agg(
        files=("file", "count"),
        subjects=("subject", "nunique"),
        sessions=("session", "nunique"),
        duration_hours=("duration_sec", lambda value: value.sum() / 3600),
        flash_events=("event_count", "sum"),
        target_flashes=("target_flash_count", "sum"),
        inferred_characters=("inferred_character_count", "sum"),
    ).reset_index()
    lines.extend(["## Coverage By Study", "", markdown_table(by_study, by_study.columns), ""])

    by_condition = file_df.groupby(["study", "phase", "condition"], dropna=False).agg(
        files=("file", "count"),
        subjects=("subject", "nunique"),
        event_count=("event_count", "sum"),
        target_flash_count=("target_flash_count", "sum"),
        inferred_character_count=("inferred_character_count", "sum"),
        median_events_per_file=("event_count", "median"),
    ).reset_index().sort_values(["study", "phase", "condition"])
    lines.extend(["## Phase And Condition Inventory", "", markdown_table(by_condition, by_condition.columns, max_rows=50), ""])

    grid_summary = file_df.groupby(["study", "grid_rows", "grid_cols"], dropna=False).agg(
        files=("file", "count"),
        grid_cell_count=("grid_cell_count", "median"),
        missing_grid_cells=("missing_grid_cells", "max"),
    ).reset_index().sort_values(["study", "grid_rows", "grid_cols"])
    lines.extend(["## Grid Layouts", "", markdown_table(grid_summary, grid_summary.columns), ""])

    lines.extend([
        "## Event Statistics",
        "",
        f"- Events per file: {compact_stats(file_df['event_count'])}",
        f"- Target flashes per file: {compact_stats(file_df['target_flash_count'])}",
        f"- Target fraction per file: {compact_stats(file_df['target_fraction'])}",
        f"- Unique stimulus codes per file: {compact_stats(file_df['unique_stimulus_codes'])}",
        "",
    ])

    if not char_df.empty:
        by_segmentation = char_df.groupby(["study", "segmentation"], dropna=False).agg(
            character_trials=("file", "count"),
            median_target_flashes=("target_flash_count", "median"),
            mean_majority_agreement=("majority_agreement", "mean"),
        ).reset_index()
        lines.extend(["## Character Trial Inference", "", markdown_table(by_segmentation, by_segmentation.columns), ""])

        top_targets = char_df.groupby(["study", "inferred_target_label"], dropna=False).size().reset_index(name="count")
        top_targets = top_targets.sort_values(["study", "count"], ascending=[True, False]).groupby("study").head(15)
        lines.extend(["## Most Frequent Inferred Targets", "", markdown_table(top_targets, top_targets.columns, max_rows=40), ""])

    if not flags_df.empty:
        flag_summary = flags_df.groupby(["study", "flag_type"], dropna=False).size().reset_index(name="count")
        flag_summary = flag_summary.sort_values(["study", "count"], ascending=[True, False])
        lines.extend(["## Quality Flags", "", markdown_table(flag_summary, flag_summary.columns, max_rows=50), ""])
        lines.append("See `quality_flags.csv` for file-level details.")
        lines.append("")
    else:
        lines.extend(["## Quality Flags", "", "No quality flags were generated.", ""])

    missing_studies = sorted(set(studies) - set(file_df["study"].unique()))
    lines.append("## Data Recommendation")
    lines.append("")
    if missing_studies:
        lines.append(
            f"Download or extract {', '.join(missing_studies)} before making StudyD/E-wide processing decisions. "
            "Current findings only reflect the studies present on disk."
        )
    else:
        lines.append(
            "StudyD and StudyE are present, so additional bigP3BCI studies are not required for the current "
            "application-specific analysis. Consider downloading more studies only for robustness experiments, "
            "domain-shift checks, or if classifier calibration remains data-limited."
        )
    lines.append("")

    lines.extend([
        "## Generated Files",
        "",
        "- `file_summary.csv`: one row per EDF file.",
        "- `character_trials.csv`: inferred target character blocks per file.",
        "- `stimulus_code_counts.csv`: stimulus code frequency distribution per file.",
        "- `grid_layouts.csv`: detected grid label, row, column, and target-code mapping.",
        "- `quality_flags.csv`: warnings that need review before downstream preprocessing.",
        "",
        "Note: StudyD `RC` fixed-condition files use 20 target flashes per inferred selection. "
        "StudyE `CB` fixed-condition files use 10 target flashes per inferred selection. "
        "Adaptive StudyD `Dyn`/`DynBigram` rows are CurrentTarget distributions, not confirmed spelled-text labels; "
        "review their flags before using them as ground truth.",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT, help="Root containing the bigP3BCI dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for report and CSV outputs.")
    parser.add_argument("--studies", nargs="+", default=["StudyD", "StudyE"], help="Study folders to scan.")
    parser.add_argument(
        "--fixed-target-flashes-per-selection",
        type=int,
        default=FALLBACK_TARGET_FLASHES_PER_SELECTION,
        help="Fallback target-flash block size for fixed-sequence character inference when no study-specific rule exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every EDF file as it is scanned.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = find_edf_files(data_root, args.studies) if data_root.exists() else []

    file_rows: List[dict] = []
    character_rows: List[dict] = []
    stimulus_rows: List[dict] = []
    grid_rows: List[dict] = []
    quality_flags: List[dict] = []

    if not data_root.exists():
        quality_flags.append({
            "study": "Unknown",
            "subject": "Unknown",
            "session": "Unknown",
            "phase": "Unknown",
            "condition": "Unknown",
            "file": "",
            "file_name": "",
            "flag_type": "missing_data_root",
            "detail": f"Data root does not exist: {data_root}",
        })

    if edf_files:
        print(f"Scanning {len(edf_files)} EDF files under {data_root}")

    for index, edf_path in enumerate(edf_files, start=1):
        if args.verbose or index == 1 or index == len(edf_files) or index % 25 == 0:
            print(f"[{index}/{len(edf_files)}] {edf_path}")
        try:
            summary, chars, stimuli, grids, flags = analyze_file(
                edf_path,
                data_root,
                args.fixed_target_flashes_per_selection,
            )
        except Exception as exc:  # Keep the scan useful even if a file is malformed.
            context = parse_context(edf_path)
            quality_flags.append(flag_row(edf_path, context, "read_error", repr(exc)))
            continue

        file_rows.append(summary)
        character_rows.extend(chars)
        stimulus_rows.extend(stimuli)
        grid_rows.extend(grids)
        quality_flags.extend(flags)

    file_df = pd.DataFrame(file_rows)
    char_df = pd.DataFrame(character_rows)
    stim_df = pd.DataFrame(stimulus_rows)
    grid_df = pd.DataFrame(grid_rows)
    flags_df = pd.DataFrame(quality_flags)

    report_path = output_dir / "bigp3bci_studyd_e_findings.md"
    write_report(report_path, data_root, args.studies, file_df, char_df, stim_df, grid_df, flags_df)
    write_csv(file_df, output_dir / "file_summary.csv")
    write_csv(char_df, output_dir / "character_trials.csv")
    write_csv(stim_df, output_dir / "stimulus_code_counts.csv")
    write_csv(grid_df, output_dir / "grid_layouts.csv")
    write_csv(flags_df, output_dir / "quality_flags.csv")

    print(f"\nWrote report: {report_path}")
    print(f"Wrote CSVs to: {output_dir}")


if __name__ == "__main__":
    main()
