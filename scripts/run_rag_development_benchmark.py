#!/usr/bin/env python3
"""Development-only chronological natural-text benchmark for the RAG stack."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor

SPLIT_PATH = ROOT / "data/rag/development_corpora/splits/public_domain_authors.json"
OUT = ROOT / "results/tables/rag_development_benchmark.json"
BANK_ROOT = ROOT / "data/rag/development_banks"
MAX_CASES = 24
TUNE_CASES = 8
CONTEXT_CHARS = 128


def phrase_entries(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def write_bank_from_training_text(path: Path, training_text: str) -> None:
    """Write entries derived only from the explicit training argument."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("text", "source", "weight", "category"))
        writer.writeheader()
        for entry in phrase_entries(training_text):
            writer.writerow({"text": entry, "source": "public_domain_development_train",
                             "weight": 1.0, "category": "development_only"})


def _key_index(char: str, grid, special_map) -> int | None:
    if char.isspace():
        label = "Sp"
    elif char.isalpha():
        label = char.upper()
    else:
        label = next((key for key, value in special_map.items() if value == char), char)
    flat = grid.reshape(-1)
    hits = np.flatnonzero(np.asarray([str(x) == label for x in flat]))
    return int(hits[0]) if len(hits) == 1 else None


def _cases(text: str, prefix: str, grid, special_map, *, limit=MAX_CASES):
    combined = prefix + text
    offset = len(prefix)
    out = []
    for i, char in enumerate(text):
        target = _key_index(char, grid, special_map)
        if target is None:
            continue
        context = combined[:offset + i][-CONTEXT_CHARS:].lower()
        out.append({"context": context, "target": target, "text_char": char})
        if len(out) >= limit:
            break
    return out


def _metrics(probabilities, targets):
    nll, ranks, top = [], [], {1: 0, 3: 0, 5: 0}
    for probs, target in zip(probabilities, targets):
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-12, None)
        probs /= probs.sum()
        order = np.argsort(-probs, kind="stable")
        rank = int(np.flatnonzero(order == target)[0]) + 1
        nll.append(float(-np.log(probs[target])))
        ranks.append(rank)
        for k in top:
            top[k] += int(rank <= k)
    count = len(targets)
    return {"n_characters": count,
            "next_character_cross_entropy_nats": float(np.mean(nll)) if nll else None,
            "top_k_accuracy": {str(k): top[k] / count for k in top} if count else {},
            "mean_true_character_rank": float(np.mean(ranks)) if ranks else None,
            "mean_reciprocal_rank": float(np.mean(1 / np.asarray(ranks))) if ranks else None,
            "keystroke_savings_top3": top[3] / count if count else None}


def _score(model, cases, base_priors):
    return [model.predict_next_char(case["context"], base_prior=base)
            if isinstance(model, RAGPredictor) else base
            for case, base in zip(cases, base_priors)]


def _candidate_gates(author, global_bank, train_phrases, tune_cases, tune_base):
    best = None
    grid = product((0.25, 0.7), (32.0, 128.0), (0.2, 0.4), (2, 5), (0.4, 0.8))
    targets = [case["target"] for case in tune_cases]
    for weight, suff_mid, subject_threshold, min_count, max_entropy in grid:
        model = RAGPredictor(
            _candidate_gates.base, phrase_bank_path=global_bank, subject_id=author,
            rag_weight=weight, retrieval_confidence_threshold=0.0,
            subject_confidence_threshold=subject_threshold,
            sufficiency_midpoint_tokens=suff_mid, sufficiency_sharpness=0.20,
            min_subject_phrases=10, bigram_min_count=min_count,
            bigram_max_normalized_entropy=max_entropy,
        )
        scores = _score(model, tune_cases, tune_base)
        loss = _metrics(scores, targets)["next_character_cross_entropy_nats"]
        config = {"rag_weight": weight, "sufficiency_midpoint_tokens": suff_mid,
                  "subject_confidence_threshold": subject_threshold,
                  "bigram_min_count": min_count,
                  "bigram_max_normalized_entropy": max_entropy,
                  "selection_metric": "training-only tune next-character cross-entropy",
                  "tune_cross_entropy": loss}
        if best is None or loss < best["tune_cross_entropy"]:
            best = config
    return best


def _make_grid():
    layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())["StudyD"]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for label, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = label
    return grid


def _online_curve(base, grid, special_map, cases, targets):
    online = RAGPredictor(base, phrase_bank_path=BANK_ROOT / "empty_global.csv",
                          subject_id="online_session", subject_only=True,
                          rag_weight=1.0, min_subject_phrases=10)
    # The smoke benchmark scores at most 24 characters per author. Use
    # thirds of the actual sample so every adaptation curve is informative.
    chunk_size = max(1, int(np.ceil(len(cases) / 3)))
    chunk_correct = {
        f"{start}-{min(start + chunk_size - 1, len(cases) - 1)}": []
        for start in range(0, len(cases), chunk_size)
    }
    word = ""
    for idx, (case, target, prior) in enumerate(zip(cases, targets, _online_curve.base_priors)):
        probs = online.predict_next_char(case["context"], base_prior=prior)
        start = (idx // chunk_size) * chunk_size
        block = f"{start}-{min(start + chunk_size - 1, len(cases) - 1)}"
        if block in chunk_correct:
            chunk_correct[block].append(int(int(np.argmax(probs)) == target))
        ch = case["text_char"]
        if ch.isspace():
            if word:
                online.update(word)
                word = ""
        else:
            word += ch
        # Teacher-forced online adaptation: the scored key is added only
        # after prediction, mimicking a successfully decoded prefix.
        if (idx + 1) % 8 == 0 and word:
            online.update(word)
            word = ""
    return {block: {"characters": len(values), "top1_accuracy": float(np.mean(values)) if values else None}
            for block, values in chunk_correct.items()}


def run():
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    grid = _make_grid()
    base = LLMPredictor(grid, local_files_only=True)
    _candidate_gates.base = base
    special_map = base.special_key_map
    BANK_ROOT.mkdir(parents=True, exist_ok=True)
    all_train = " ".join(v["train_text"] for v in payload["authors"].values())
    global_bank = BANK_ROOT / "phrase_bank_global.csv"
    write_bank_from_training_text(global_bank, all_train)
    for author, data in payload["authors"].items():
        write_bank_from_training_text(BANK_ROOT / "by_subject" / f"phrase_bank_{author}.csv", data["train_text"])

    author_results, author_tests = {}, {}
    for author, data in payload["authors"].items():
        train_text, tune_text, eval_text = data["train_text"], data["tune_text"], data["evaluation_text"]
        tune_cases = _cases(tune_text, train_text, grid, special_map, limit=TUNE_CASES)
        tune_base = [base.predict_next_char(c["context"]) for c in tune_cases]
        gates = _candidate_gates(author, global_bank, phrase_entries(train_text), tune_cases, tune_base)
        # Held-out development text is first scored after training-only gate selection.
        test_prefix = train_text + " " + tune_text
        test_cases = _cases(eval_text, test_prefix, grid, special_map, limit=MAX_CASES)
        test_base = [base.predict_next_char(c["context"]) for c in test_cases]
        targets = [case["target"] for case in test_cases]
        modes = {
            "LM": test_base,
            "LM+global_bank": _score(RAGPredictor(
                base, phrase_bank_path=global_bank, rag_weight=0.7,
                retrieval_confidence_threshold=0.0), test_cases, test_base),
            "LM+subject_bank": _score(RAGPredictor(
                base, phrase_bank_path=global_bank, subject_id=author, subject_only=True,
                rag_weight=1.0, min_subject_phrases=10), test_cases, test_base),
            "gated_blend": _score(RAGPredictor(
                base, phrase_bank_path=global_bank, subject_id=author,
                rag_weight=gates["rag_weight"], retrieval_confidence_threshold=0.0,
                subject_confidence_threshold=gates["subject_confidence_threshold"],
                sufficiency_midpoint_tokens=gates["sufficiency_midpoint_tokens"],
                sufficiency_sharpness=0.20, min_subject_phrases=10,
                bigram_min_count=gates["bigram_min_count"],
                bigram_max_normalized_entropy=gates["bigram_max_normalized_entropy"],
            ), test_cases, test_base),
        }
        author_results[author] = {"train_words": data["train_words"], "tune_words": data["tune_words"],
                                  "test_words": data["evaluation_words"], "test_characters": len(targets),
                                  "training_only_fitted_gates": gates,
                                  "metrics": {mode: _metrics(scores, targets) for mode, scores in modes.items()}}
        author_tests[author] = (data, test_cases, test_base, targets)

    # Author-level macro summaries, with no participant interpretation.
    modes = ["LM", "LM+global_bank", "LM+subject_bank", "gated_blend"]
    macro = {}
    for mode in modes:
        macro[mode] = {
            metric: float(np.mean([author_results[a]["metrics"][mode][metric]
                                   for a in author_results]))
            for metric in ("next_character_cross_entropy_nats", "mean_true_character_rank",
                           "mean_reciprocal_rank", "keystroke_savings_top3")
        }
        for k in (1, 3, 5):
            macro[mode].setdefault("top_k_accuracy", {})[str(k)] = float(np.mean([
                author_results[a]["metrics"][mode]["top_k_accuracy"][str(k)] for a in author_results]))

    # Learning curves use chronological prefixes of each author's training text.
    learning = {}
    for size in (0, 64, 128, 256, 512, 1024):
        author_scores = []
        size_root = BANK_ROOT / f"learning_{size}"
        size_root.mkdir(parents=True, exist_ok=True)
        global_prefix_texts = []
        per_author_prefix = {}
        for author, data in payload["authors"].items():
            tokens = data["train_text"].split()
            prefix = " ".join(tokens[:size])
            per_author_prefix[author] = prefix
            global_prefix_texts.append(prefix)
        size_global = size_root / "phrase_bank_global.csv"
        write_bank_from_training_text(size_global, " ".join(global_prefix_texts))
        for author, data in payload["authors"].items():
            prefix = per_author_prefix[author]
            size_subject = size_root / "by_subject" / f"phrase_bank_{author}.csv"
            write_bank_from_training_text(size_subject, prefix)
            cases = author_tests[author][1]
            bases = author_tests[author][2]
            targets = author_tests[author][3]
            subject_only = RAGPredictor(base, phrase_bank_path=size_global, subject_id=author,
                                        subject_only=True, rag_weight=1.0, min_subject_phrases=1)
            gated = RAGPredictor(base, phrase_bank_path=size_global, subject_id=author,
                                 rag_weight=0.7, retrieval_confidence_threshold=0.0,
                                 min_subject_phrases=1)
            global_model = RAGPredictor(base, phrase_bank_path=size_global,
                                        rag_weight=0.7, retrieval_confidence_threshold=0.0)
            author_scores.append({"author": author,
                                  "subject_bank": _metrics(_score(subject_only, cases, bases), targets),
                                  "global_bank": _metrics(_score(global_model, cases, bases), targets),
                                  "gated_blend": _metrics(_score(gated, cases, bases), targets)})
        learning[str(size)] = author_scores

    online = {}
    for author, (data, cases, bases, targets) in author_tests.items():
        _online_curve.base_priors = bases
        online[author] = _online_curve(base, grid, special_map, cases, targets)

    # Cross-author topic switch: Jane Austen training bank, Moby-Dick held-out text.
    switch_author = "Herman_Melville"
    switch_data = payload["authors"][switch_author]
    switch_cases = _cases(switch_data["evaluation_text"], switch_data["tune_text"], grid,
                          special_map, limit=MAX_CASES)
    switch_base = [base.predict_next_char(c["context"]) for c in switch_cases]
    austen_bank = BANK_ROOT / "by_subject" / "phrase_bank_Jane_Austen.csv"
    switch_model = RAGPredictor(base, phrase_bank_path=global_bank, subject_id="Jane_Austen",
                                subject_only=True, rag_weight=1.0)
    switch = {"label": "development-only cross-author topic switch; not participant evidence",
              "from": "Jane Austen training bank", "to": "Herman Melville held-out text",
              "metrics": {"LM": _metrics(switch_base, [c["target"] for c in switch_cases]),
                          "subject_bank": _metrics(_score(switch_model, switch_cases, switch_base),
                                                   [c["target"] for c in switch_cases])}}

    result = {
        "status": "development_only_completed",
        "simulated_or_measured": "offline public-domain text benchmark; no participant data",
        "split_sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
        "evaluation_text_entered_bank": False,
        "base_context_chars": CONTEXT_CHARS,
        "max_scored_cases_per_author": MAX_CASES,
        "models": modes,
        "author_results": author_results,
        "author_macro_metrics": macro,
        "learning_curve_by_training_words": learning,
        "online_coldstart_personalization_curves": online,
        "topic_switch": switch,
        "gate_tuning_scope": "chronological middle block from each author only; final block untouched until scoring",
        "claim_warning": "All outputs debug the offline code path. They are not evidence of participant personalization benefit.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    output = run()
    print(json.dumps({"status": output["status"], "authors": list(output["author_results"]),
                      "macro": output["author_macro_metrics"],
                      "evaluation_text_entered_bank": output["evaluation_text_entered_bank"]}, indent=2))
