# P300-LLM Speller

A P300 event-related-potential (ERP) speller that fuses classifier posteriors with a
lightweight language model, extending the Bayesian dynamic-stopping / bigram-language-model
line of work (Mainsah, Throckmorton, et al.) with a modern LLM-based prior. Solo capstone
project; secondary goal is an academic publication (arXiv preprint → workshop-tier venue).

- **Repo:** https://github.com/shivankbhatia/capstone_project
- **Active branch:** `SKB-J8`
- **Dataset:** [bigP3BCI](https://physionet.org/) on PhysioNet — Study D is the primary
  benchmark (307 EDF files, 9×8 grid = 72 character classes). Study D was chosen because it
  explicitly includes predictive-spelling / dynamic-bigram-stopping conditions, making it a
  natural testbed for extending the Mainsah/Throckmorton baseline.
  (BNCI2014-008 was the original dataset choice, abandoned due to server unreliability.)

---

## 1. Project goal

Classic P300 spellers decode which grid cell a user attended to purely from EEG evidence
(row/column flash classifier posteriors). This project fuses those posteriors, in log
domain, with a character-level prior from a language model — testing whether a modern LLM
prior improves accuracy/ITR (information transfer rate) over EEG evidence alone, and how
that compares to the classical bigram/HMM priors used in prior P300-speller literature.

Three-arm ablation design (built into the pipeline from day one):
1. **Baseline** — EEG classifier only, no LM fusion
2. **Fixed-weight fusion** — Bayesian fusion, constant weighting of the LM prior
3. **Adaptive fusion** — Bayesian fusion, LM-prior weight scales with EEG posterior entropy
   (trust the LM more when the EEG evidence is uncertain)

A fourth arm (retrieval-augmented / RAG-LLM prior, personalized to a phrase corpus) is
planned but currently deferred — see [Known gaps](#7-known-gaps--open-questions).

---

## 2. Repository structure (actual, as of this document)

```
capstone_project/
├── run_pipeline.py                  # Root-level: the REAL end-to-end pipeline
├── requirements.txt
├── setup_env.sh
├── data/
│   ├── raw/                         # gitignored — bigP3BCI EDF files live here locally
│   └── processed/                   # gitignored — generated .fif epochs, JSON registries
├── src/
│   ├── data/
│   │   ├── batch_preprocess.py      # Full-corpus EDF → epoched .fif batch runner
│   │   ├── load_bigp3bci.py
│   │   └── load_bnci2014008.py      # legacy, dataset since abandoned
│   ├── preprocessing/
│   │   ├── epoching.py
│   │   └── build_session_sequence.py  # Ground-truth registry → per-character trial sequences
│   ├── models/
│   │   ├── classifier.py            # Calibrated LDA / SWLDA-style P300 classifier
│   │   ├── decoder.py                # P300Decoder — evidence accumulation, Wolpaw ITR
│   │   ├── fusion.py                 # BayesianFusionEngine — the core novelty
│   │   └── llm_predictor.py          # LLMPredictor — distilGPT2 next-char prior
│   └── evaluation/
│       ├── metrics.py                # SpellerMetrics, AblationTracker
│       ├── ablations.py              # stub — not yet implemented (see gaps)
│       └── baseline_eval.py          # STALE/dead — superseded by root run_pipeline.py
├── scripts/
│   ├── analyze_bigp3bci_studyd_e.py  # Study D exploratory analysis
│   └── run_pipeline.py               # dead stub duplicate — DO NOT USE, see gaps
└── results/
    ├── bigp3bci_studyd_e_findings/   # grid_layouts.csv, file_summary.csv, etc.
    └── threshold_sweep.png
```

**Naming note:** an earlier implementation plan referenced `edf_parser.py` /
`segmentation.py` / `src/evaluation/run_pipeline.py` — those names were never used; the
actual files are as listed above.

---

## 3. Pipeline stages — status

| Stage | File(s) | Status |
|---|---|---|
| EDF parsing & channel classification | `batch_preprocess.py` | ✅ Complete |
| Grid-layout extraction | `batch_preprocess.py` | ✅ Complete — confirmed constant 9×8 across all 307 Study D files |
| Ground-truth segmentation (RC/Train, fixed-block) | `batch_preprocess.py` | ✅ Complete |
| Ground-truth segmentation (Dyn/DynBigram, adaptive) | `batch_preprocess.py` | ✅ Complete — see [history](#6-debugging-history--why-this-mattered), two real bugs found and fixed |
| `StimulusCode` → `epochs.metadata` preservation | `batch_preprocess.py`, `epoching.py` | ✅ Complete |
| Per-character flash indexing for adaptive files | `run_pipeline.py`, `batch_preprocess.py` | ✅ Complete — metadata-driven (`char_index`, `sequence_in_char`), fixed-block logic is fallback-only |
| Session sequence building | `build_session_sequence.py` | ✅ Complete |
| P300 classifier (SWLDA-style) | `classifier.py` | ✅ Implemented — not yet trained on the corrected full corpus |
| Character decoder | `decoder.py` | ✅ Complete — dynamic `num_classes`, Wolpaw ITR |
| Fusion engine | `fusion.py` | ✅ Complete — fixed & adaptive-entropy modes, numerically stable log-domain math |
| LLM prior | `llm_predictor.py` | ✅ Complete, but simplified — see [gaps](#7-known-gaps--open-questions) |
| End-to-end pipeline wiring | `run_pipeline.py` (root) | ✅ Complete — real 3-arm ablation harness |
| Metrics / ablation tracking | `metrics.py` | ✅ Complete |
| Formal ablation runner | `ablations.py` | ❌ Stub — logic currently lives inline in `run_pipeline.py` instead |
| Statistical significance testing | — | ❌ Not started |
| Classifier trained on corrected data | — | ❌ Not started — **next concrete step** |
| Baseline (No-LLM) results | — | ❌ Not started |
| LLM-fusion results | — | ❌ Not started |
| RAG/retrieval layer | — | ⏸️ Deferred, previously existed then was dropped in a rewrite |

---

## 4. Environment

- Python 3.9.6, project virtual environment at `.venv`
- Key packages: `mne` (1.8.0), `moabb` (1.2.0), `numpy` (1.26.4), `torch` (2.8.0),
  `transformers` (distilGPT2), `scikit-learn`
- scikit-learn import issue (previously blocking SWLDA/sklearn work) — **resolved**
- `pip install` requires `--break-system-packages` in this environment

---

## 5. Core methodology notes

**Grid-index encoding:** `idx = (row - 1) * n_cols + col` (column is *not* shifted by -1).
Confirmed against user-verified ground truth (`T_4_2→20`, `H_2_2→8`, `E_1_5→5` matches the
spelled word "THE"). Index 0 is unreachable under this formula, so `CurrentTarget == 0`
means "no active target" (idle/reset), not a real character.

**Grid layout:** confirmed constant (9 rows × 8 cols, 72 cells, 0 missing) across all 307
Study D files via `analyze_bigp3bci_studyd_e.py` — ruled out a per-session-randomized-layout
hypothesis that was raised and investigated during development.

**Fixed-block segmentation (RC/Train, calibration files):** `sequences_per_selection = 20`
— confirmed empirically correct for these files.

**Adaptive segmentation (Dyn/DynBigram, test files):** ground truth is read directly from
the `SelectedTarget` channel, which pulses from `0` to a nonzero code exactly once per
finalized character selection — regardless of whether consecutive selections share the same
code (i.e., correctly handles doubled letters). This is the final, validated approach after
two earlier approaches were tried and superseded — see below.

**Fusion math:** log-domain combination of EEG log-posterior and α-weighted LLM
log-prior, normalized via `logsumexp` for numerical stability. In adaptive mode, α scales
with the normalized entropy of the EEG posterior (trust the LM more when EEG is uncertain).

---

## 6. Debugging history — why this mattered

This project surfaced several real correctness bugs during preprocessing, each of which
would have silently corrupted downstream results (classifier training, fusion, baseline
metrics) if left unnoticed. Documenting them here because they're relevant to the paper's
methods section (data quality / ground-truth construction) as well as to anyone continuing
this work.

### 6.1 Dyn/DynBigram whole-file majority vote (wrong assumption)
**Original approach:** treat each Dyn/DynBigram file as one character trial, decode via a
single whole-file majority vote over `CurrentTarget`.
**Symptom:** every single adaptive file showed low vote agreement (18–39%), with vote
distributions showing multiple comparable-sized clusters rather than one dominant code —
not noise, a systematic pattern.
**Root cause:** `CurrentTarget` changes value multiple times within a file — these files
contain *multiple* character selections, not one.
**Fix (interim):** `segment_by_target_transitions()` — split into contiguous runs of
constant `CurrentTarget`, vote within each run.
**Result:** per-run agreement jumped to ~100%, character counts matched real word lengths
(e.g. "VISUAL" → 6 runs).

### 6.2 Doubled letters silently dropped (interim fix's blind spot)
**Symptom:** spot-checking decoded strings against real English words revealed systematic
undercounts — `BUTTON → BUTON`, `HIDDEN → HIDEN`, `ALLOWS → ALOWS`, `INDEED → INDED` — every
case losing exactly one letter, always at a doubled-letter position.
**Root cause:** run-transition segmentation can only detect a new character when the
*code changes*. Two consecutive selections of the *same* code (a doubled letter) have no
detectable boundary between them under this method — and the resulting "100% run
agreement" metric is tautologically guaranteed by construction, so it couldn't self-detect
this failure mode.
**Fix (final):** switched to direct pulse-detection on the `SelectedTarget` channel, which
the recording system pulses once per *finalized* character selection — an independent
signal, not inferred from flash-code repetition. Confirmed via raw-channel inspection
(`SelectedTarget`, `SelectedRow`, `SelectedColumn` all show 12 changes = 6 pulses for a
6-letter word; codes cross-validated against `SelectedRow`/`SelectedColumn` via the known
`idx = (row-1)*n_cols + col` formula).
**Verified:** BUTTON, HIDDEN, ALLOWS all decode correctly (doubled letters captured);
numeric target sequences (e.g. `598524`) still decode correctly through the same path.
**Full rerun:** 307/307 files processed, 0 failures.

### 6.3 Grid-layout mismatch false positives
**Symptom:** 205/205 adaptive files flagged as "grid layout mismatch" against the canonical
layout.
**Root cause:** the comparison checked live-parsed `grid_map` (Python tuples) against
`grid_layout.json` loaded from disk (JSON has no tuple type, so coordinates round-trip as
lists) — comparing types, not values.
**Fix:** normalize coordinates to lists on both sides before comparing; restored the check
to a hard failure (not a logged-and-continue quality flag) since the underlying layout truly
is constant and any real future mismatch should stop the run, not slide through unnoticed.

### 6.4 Metadata-driven flash indexing (pipeline-layer follow-up)
After fixing ground truth at the preprocessing layer, `run_pipeline.py`'s flash-indexing
logic still assumed fixed-size blocks (`char_idx * flashes_per_char`), which would
mis-slice adaptive files even with correct ground truth upstream. Fixed by adding
`char_index` / `sequence_in_char` metadata columns to adaptive epochs (derived from
`SelectedTarget` pulse boundaries) and switching `run_pipeline.py` to metadata-driven
indexing for adaptive files, with fixed-block logic retained as a fallback (with a visible
warning) for any `.fif` file generated before this change.
**Verified:** spot-check on the BUTTON file confirmed the T/T boundary is correctly
resolved — flashes for the first T end before its completion pulse, flashes for the second T
begin after it and before its own completion pulse.

---

## 7. Known gaps / open questions

- **RAG/retrieval layer is currently absent.** An earlier version of `llm_predictor.py`
  (`LanguagePriorEngine`) included TF-IDF retrieval against a phrase corpus, fused with the
  LM prior. The current `LLMPredictor` is a clean, grid-agnostic distilGPT2 next-char
  predictor only — no retrieval component. Since "RAG-LLM" is central to the project's
  novelty framing, this needs an explicit decision: was dropping it intentional
  (baseline-first sequencing — pure LLM now, retrieval as a later ablation arm), and if so,
  when does it get reintroduced?
- **`ablations.py` is an empty stub.** The actual ablation-tracking logic currently lives
  inline in `run_pipeline.py` (`SimpleAblationTracker`), duplicating what `metrics.py`'s
  `AblationTracker` class is meant to do. Needs consolidation.
- **`baseline_eval.py` is dead code.** Calls a `P300Decoder` API
  (`decode_character(char_codes, char_probs)`, `.n_rows`, `.calculate_itr()` as an instance
  method) that no longer matches the current `decoder.py`. Superseded by root
  `run_pipeline.py`; should be deleted or rewritten.
- **`scripts/run_pipeline.py` is a dead stub** (`raise NotImplementedError`), a leftover
  duplicate of the real pipeline at repo root. Should be deleted to avoid confusion.
- **One adaptive file (`D_04_SE001_Dyn_Test01`, decoded `INDEEE`) has unverified ground
  truth.** Raw `SelectedTarget` signal is unambiguous (solid, non-jittery pulses), so the
  pulse-detection mechanism is reading the channel correctly — but whether `INDEEE` is the
  true intended target, a genuine participant selection error, or something else is unverified
  against any external reference. Since 3/4 spot-checked files independently confirm the
  detection method works, this was set aside rather than blocking progress — worth resolving
  if dataset documentation surfaces a target-phrase reference.
- **Classifier has not yet been trained on the corrected full corpus.** All fusion/decoder
  work has been validated structurally, but no real accuracy/ITR numbers exist yet.

---

## 8. Reproducing the current state

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --break-system-packages

# 2. Preprocess the full Study D corpus (resume-safe; skips already-processed files)
python src/data/batch_preprocess.py
# Expect: Processed: 307, Failed: 0, 0 grid-layout mismatches

# 3. (Next) Train the classifier
python src/models/classifier.py

# 4. (Next) Run the baseline / fusion ablation pipeline
python run_pipeline.py
```

---

## 9. References

- Mainsah, B.O., Throckmorton, C.S., et al. — Bayesian dynamic-stopping and bigram/HMM
  language-model work for P300 spellers (this project's baseline being extended)
- bigP3BCI dataset, PhysioNet
