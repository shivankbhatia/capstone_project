#!/usr/bin/env python3
"""Prepare a local, anonymized, chronological RAG text sample.

This utility is intentionally not run on participant data in this repository.
It writes the evaluation tail separately and constructs the live phrase bank
from the first 60% only. Name redaction remains a participant-reviewed step.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
SENTENCES = re.compile(r"(?<=[.!?])\s+")


def anonymize(text: str) -> str:
    text = EMAIL.sub(" [EMAIL] ", text)
    return PHONE.sub(" [PHONE] ", text)


def read_sample(path: Path) -> str:
    if path.suffix.lower() != ".jsonl":
        return path.read_text(encoding="utf-8")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        records.append((str(item.get("timestamp", "")), str(item["text"])))
    if records and all(timestamp for timestamp, _ in records):
        records.sort(key=lambda pair: pair[0])
    return "\n".join(text for _, text in records)


def split_text(text: str, min_tokens: int = 700) -> dict:
    clean = anonymize(text).strip()
    tokens = clean.split()
    if len(tokens) < min_tokens:
        raise ValueError(f"sample has {len(tokens)} tokens; requires at least {min_tokens}")
    first = int(len(tokens) * 0.6)
    second = int(len(tokens) * 0.8)
    return {
        "bank_training_text": " ".join(tokens[:first]),
        "gate_tuning_text": " ".join(tokens[first:second]),
        "evaluation_text": " ".join(tokens[second:]),
        "token_counts": {"total": len(tokens), "bank_training": first,
                         "gate_tuning": second - first, "evaluation": len(tokens) - second},
        "split": "chronological 60/20/20; evaluation text excluded from bank",
    }


def build_bank(bank_path: Path, training_text: str) -> None:
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    with bank_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "source", "weight", "category"))
        writer.writeheader()
        for sentence in SENTENCES.split(training_text):
            sentence = sentence.strip()
            if sentence:
                writer.writerow({"text": sentence, "source": "participant_training",
                                 "weight": 1.0, "category": "personal"})


def prepare(source: Path, output_dir: Path, min_tokens: int = 700) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = split_text(read_sample(source), min_tokens=min_tokens)
    (output_dir / "splits.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    build_bank(output_dir / "phrase_bank_global.csv", payload["bank_training_text"])
    return payload["token_counts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local UTF-8 text or timestamped JSONL")
    parser.add_argument("output_dir", type=Path, help="local-only output directory")
    parser.add_argument("--min-tokens", type=int, default=700)
    args = parser.parse_args()
    counts = prepare(args.source, args.output_dir, args.min_tokens)
    print(json.dumps({"prepared": True, "counts": counts, "text_logged": False}))


if __name__ == "__main__":
    main()
