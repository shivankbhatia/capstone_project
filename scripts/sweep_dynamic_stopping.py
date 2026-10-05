#!/usr/bin/env python3
"""Evaluate standard row/column replay stopping rules on held-out sessions.

The no-argument configuration is the selected policy: tau=0.80, a two-step
minimum, a 10-step cap, and characters with at least one recorded sequence.
Other grid points remain available for audit and exploratory reruns.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.server import ROOT, ReplayService


THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
RECOMMENDED_THRESHOLD = 0.80
RECOMMENDED_MIN_SEQUENCES = 2
RECOMMENDED_MAX_SEQUENCES = 10
def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    selections = [selection for result in results for selection in result["selections"]]
    count = len(selections)
    correct = sum(selection["correct"] for selection in selections)
    sequences = [len(selection["trace"]) for selection in selections]
    histogram = {
        "correct": dict(sorted(Counter(
            len(selection["trace"]) for selection in selections if selection["correct"]
        ).items())),
        "incorrect": dict(sorted(Counter(
            len(selection["trace"]) for selection in selections if not selection["correct"]
        ).items())),
    }
    return {
        "characters": count,
        "mean_sequences_per_character": sum(sequences) / count if count else 0.0,
        "mean_flashes_per_character": sum(
            selection.get("flashes_used", 0) for selection in selections
        ) / count if count else 0.0,
        "accuracy": 100 * correct / count if count else 0.0,
        "stopping_point_histogram": histogram,
    }


def empty_aggregate() -> dict[str, Any]:
    return {
        "characters": 0,
        "correct": 0,
        "sequences": 0,
        "flashes": 0,
        "correct_stops": Counter(),
        "incorrect_stops": Counter(),
    }


def usable_character_keys(service: ReplayService, minimum_sequences: int = 1) -> set[tuple[str, int]]:
    """Return characters with at least one recorded row/column flash.

    Partial sequences count as coverage. Characters with no usable flash are
    excluded from stopping-policy evaluation because they test only the prior.
    """
    usable = set()
    for sample in service.samples():
        session_id = sample["id"]
        metadata = mne.read_epochs(
            ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif",
            preload=False, verbose=False,
        ).metadata
        adaptive = (
            metadata is not None
            and {"char_index", "sequence_in_char"}.issubset(metadata.columns)
        )
        if adaptive:
            valid = metadata[
                (metadata["char_index"] >= 0)
                & (metadata["sequence_in_char"] > 0)
                & (metadata["stimulus_code"] >= 1)
                & (metadata["stimulus_code"] <= 17)
            ]
            depth = valid.groupby("char_index")["sequence_in_char"].nunique()
            usable.update(
                (session_id, int(index))
                for index, count in depth.items()
                if count >= minimum_sequences
            )
        else:
            # The legacy reader uses 15 fixed 17-flash blocks per character.
            for index in range(len(sample["target"])):
                start = index * 15 * 17
                block = metadata.iloc[start:start + 15 * 17]
                depth = sum(
                    ((block.iloc[sequence * 17:(sequence + 1) * 17]["stimulus_code"] >= 1)
                    & (block.iloc[sequence * 17:(sequence + 1) * 17]["stimulus_code"] <= 17)).any()
                    for sequence in range(15)
                )
                if depth >= minimum_sequences:
                    usable.add((session_id, index))
    return usable


def add_result(aggregate: dict[str, Any], result: dict[str, Any],
               usable: set[tuple[str, int]]) -> None:
    for selection in result["selections"]:
        if (result["session_id"], selection["index"]) not in usable:
            continue
        used = len(selection["trace"])
        aggregate["characters"] += 1
        aggregate["correct"] += int(selection["correct"])
        aggregate["sequences"] += used
        aggregate["flashes"] += selection.get("flashes_used", 0)
        aggregate["correct_stops" if selection["correct"] else "incorrect_stops"][used] += 1


def finalize_aggregate(aggregate: dict[str, Any]) -> dict[str, Any]:
    count = aggregate["characters"]
    sequences = aggregate["sequences"]
    return {
        "characters": count,
        "mean_sequences_per_character": sequences / count if count else 0.0,
        "mean_flashes_per_character": aggregate["flashes"] / count if count else 0.0,
        "accuracy": 100 * aggregate["correct"] / count if count else 0.0,
        "stopping_point_histogram": {
            "correct": dict(sorted(aggregate["correct_stops"].items())),
            "incorrect": dict(sorted(aggregate["incorrect_stops"].items())),
        },
    }


def archive_current_arms() -> None:
    """Persist the pre-tuning standard and compact replay observations."""
    service = ReplayService()
    session_ids = [sample["id"] for sample in service.samples()]
    compact = [service.replay(session_id, scan_mode="compact") for session_id in session_ids]
    standard = [service.replay(session_id, scan_mode="standard") for session_id in session_ids]
    output_dir = ROOT / "dead_arms" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "focus_scan_flash_savings.json").write_text(
        json.dumps({"scan_mode": "compact", "sessions": compact, "aggregate": summarize(compact)}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "standard_interleaved_rc_summary.json").write_text(
        json.dumps({"scan_mode": "standard", "sessions": standard, "aggregate": summarize(standard)}, indent=2),
        encoding="utf-8",
    )


def plot_pareto(results: list[dict[str, Any]]) -> None:
    """Plot the threshold sweep with archived comparison arms."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4.5))
    default_rows = [
        row for row in results
        if row["coverage_min_sequences"] == 1
        and row["min_sequences"] == 2 and row["max_sequences"] == 15
    ]
    axis.plot(
        [row["mean_flashes_per_character"] for row in default_rows],
        [row["accuracy"] for row in default_rows],
        "o-", color="#1f77b4", label="Threshold sweep (min=2, max=15)",
    )
    refinement_rows = [
        row for row in results
        if row["coverage_min_sequences"] == 1 and row not in default_rows
    ]
    axis.scatter(
        [row["mean_flashes_per_character"] for row in refinement_rows],
        [row["accuracy"] for row in refinement_rows],
        color="#4c9f70", label="Sequence-bound refinement",
    )
    for row in [row for row in results if row["coverage_min_sequences"] == 1]:
        axis.annotate(
            f"τ={row['confidence_threshold']:.2f}, {row['min_sequences']}/{row['max_sequences']}",
            (row["mean_flashes_per_character"], row["accuracy"]),
            xytext=(4, 4), textcoords="offset points", fontsize=7,
        )

    standard_default = next(
        (row for row in default_rows if row["confidence_threshold"] == 0.85), None
    )
    if standard_default is not None:
        axis.scatter(standard_default["mean_flashes_per_character"], standard_default["accuracy"],
                     marker="s", color="#555555", label="Standard default (τ=0.85)")
    # This is the archived row-focus comparison point reported with the
    # compact-scan experiment; it uses its experiment-level flash total.
    axis.scatter(117.3, 50.6, marker="X", s=80, color="#b22222",
                 label="Row-focus compact (dead arm: 117.3 / 50.6%)")

    axis.set_xlabel("Mean flashes per character")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Dynamic stopping: flash/accuracy trade-off")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output = ROOT / "results" / "figures" / "dynamic_stopping_threshold_pareto.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_threshold_sweep(thresholds: tuple[float, ...] = (RECOMMENDED_THRESHOLD,),
                        min_sequences: int = RECOMMENDED_MIN_SEQUENCES,
                        max_sequences: int = RECOMMENDED_MAX_SEQUENCES,
                        coverage_min_sequences: int = 1) -> list[dict[str, Any]]:
    """Run one threshold slice of the dynamic stopping grid."""
    service = ReplayService()
    session_ids = [sample["id"] for sample in service.samples()]
    usable = usable_character_keys(service, coverage_min_sequences)
    print(
        f"Evaluating {len(usable)} characters with at least "
        f"{coverage_min_sequences} recorded sequence(s)."
    )
    aggregates = {threshold: empty_aggregate() for threshold in thresholds}
    # Keep each session contiguous so run_pipeline's epoch cache is reused
    # across all threshold settings.
    for session_id in session_ids:
        for threshold in thresholds:
            result = service.replay(
                session_id,
                scan_mode="standard",
                confidence_threshold=threshold,
                min_sequences=min_sequences,
                max_sequences=max_sequences,
            )
            add_result(aggregates[threshold], result, usable)
            # The full flash traces are useful for one replay response, but a
            # threshold sweep needs only its aggregate diagnostics.
            service._replay_cache.clear()
            del result
            gc.collect()
    rows = []
    for threshold in thresholds:
        row = {
            "confidence_threshold": threshold,
            "min_sequences": min_sequences,
            "max_sequences": max_sequences,
            "coverage_min_sequences": coverage_min_sequences,
            **finalize_aggregate(aggregates[threshold]),
        }
        rows.append(row)
        print(
            f"threshold={threshold:.2f} accuracy={row['accuracy']:.2f}% "
            f"sequences/char={row['mean_sequences_per_character']:.2f} "
            f"flashes/char={row['mean_flashes_per_character']:.2f}"
        )
    output = ROOT / "results" / "tables" / "dynamic_stopping_threshold_sweep.json"
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    for row in existing:
        row.setdefault("coverage_min_sequences", 1)
    current_keys = {
        (threshold, min_sequences, max_sequences, coverage_min_sequences)
        for threshold in thresholds
    }
    untouched = [
        row for row in existing
        if (row["confidence_threshold"], row["min_sequences"], row["max_sequences"],
            row["coverage_min_sequences"])
        not in current_keys
    ]
    combined = sorted(
        untouched + rows,
        key=lambda row: (row["coverage_min_sequences"], row["confidence_threshold"],
                         row["min_sequences"], row["max_sequences"]),
    )
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    plot_pareto(combined)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-current", action="store_true")
    parser.add_argument("--threshold", type=float, default=RECOMMENDED_THRESHOLD,
                        choices=THRESHOLDS, help="Threshold to run.")
    parser.add_argument("--min-sequences", type=int, default=RECOMMENDED_MIN_SEQUENCES,
                        choices=(1, 2, 3))
    parser.add_argument("--max-sequences", type=int, default=RECOMMENDED_MAX_SEQUENCES,
                        choices=(10, 15, 20))
    parser.add_argument("--coverage-min-sequences", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--plot-only", action="store_true",
                        help="Render the Pareto plot from the saved table.")
    args = parser.parse_args()
    if args.archive_current:
        archive_current_arms()
        return
    if args.plot_only:
        output = ROOT / "results" / "tables" / "dynamic_stopping_threshold_sweep.json"
        plot_pareto(json.loads(output.read_text(encoding="utf-8")))
        return
    run_threshold_sweep(
        (args.threshold,),
        min_sequences=args.min_sequences,
        max_sequences=args.max_sequences,
        coverage_min_sequences=args.coverage_min_sequences,
    )


if __name__ == "__main__":
    main()
