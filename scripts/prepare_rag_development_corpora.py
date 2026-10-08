#!/usr/bin/env python3
"""Create author-separated, chronological public-domain RAG dev splits.

These texts are for pipeline debugging only, never participant claims.
The split is word-token based so it remains stable across tokenizer versions.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/rag/development_corpora/source"
SPLITS = ROOT / "data/rag/development_corpora/splits"
AUTHOR_FILES = {
    "Jane_Austen": ("pride_and_prejudice_pg1342.txt", "Pride and Prejudice", "Project Gutenberg eBook 1342", "https://www.gutenberg.org/ebooks/1342"),
    "Mary_Shelley": ("frankenstein_pg84.txt", "Frankenstein", "Project Gutenberg eBook 84", "https://www.gutenberg.org/ebooks/84"),
    "Herman_Melville": ("moby_dick_pg15.txt", "Moby-Dick", "Project Gutenberg eBook 15", "https://www.gutenberg.org/ebooks/15"),
    "Arthur_Conan_Doyle": ("adventures_sherlock_excerpt_local.txt", "The Adventures of Sherlock Holmes excerpt", "Local public-domain excerpt, existing project development fixture", "https://www.gutenberg.org/ebooks/1661"),
}


def clean_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    if start:
        text = text[start.end():]
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*", text, re.I)
    if end:
        text = text[:end.start()]
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", "-")
    text = text.replace("\u2026", "...")
    return " ".join(text.split())


def build(max_words: int = 4000, train_fraction: float = 0.6, tune_fraction: float = 0.2,
          output_path: Optional[Path] = None) -> dict:
    if max_words < 100 or not 0.5 <= train_fraction < 1 or tune_fraction <= 0 or train_fraction + tune_fraction >= 1:
        raise ValueError("Require max_words >= 100 and valid chronological train/tune/test fractions")
    SPLITS.mkdir(parents=True, exist_ok=True)
    authors = {}
    for author, (filename, title, source_name, url) in AUTHOR_FILES.items():
        source_path = SOURCE / filename
        raw_bytes = source_path.read_bytes()
        tokens = clean_text(raw_bytes.decode("utf-8", errors="replace")).split()
        tokens = tokens[:max_words]
        train_cut = int(len(tokens) * train_fraction)
        tune_cut = int(len(tokens) * (train_fraction + tune_fraction))
        train, tune, evaluation = tokens[:train_cut], tokens[train_cut:tune_cut], tokens[tune_cut:]
        if not train or not tune or not evaluation:
            raise ValueError(f"Empty chronological split for {author}")
        authors[author] = {
            "title": title, "source": source_name, "source_url": url,
            "source_path": str(source_path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "token_definition": "whitespace-delimited tokens after header/footer and whitespace normalization",
            "total_words": len(tokens), "train_words": len(train), "tune_words": len(tune), "evaluation_words": len(evaluation),
            "train_text": " ".join(train), "tune_text": " ".join(tune), "evaluation_text": " ".join(evaluation),
        }
    payload = {
        "development_only": True,
        "participant_or_clinical_claims_allowed": False,
        "split": {"method": "chronological prefix, middle gate-tuning block, final evaluation block",
                  "bank_training_fraction": train_fraction, "gate_tuning_fraction": tune_fraction,
                  "evaluation_fraction": 1 - train_fraction - tune_fraction,
                  "max_words_per_author": max_words},
        "authors": authors,
    }
    out = output_path or (SPLITS / "public_domain_authors.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"development_only": True, "split_path": str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out),
                "split_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "authors": {a: {k: v for k, v in data.items() if k not in {"train_text", "tune_text", "evaluation_text"}}
                            for a, data in authors.items()}}
    if output_path is None:
        (SPLITS / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({a: {"train_words": v["train_words"], "tune_words": v["tune_words"], "evaluation_words": v["evaluation_words"]}
                      for a, v in result["authors"].items()}, indent=2))
