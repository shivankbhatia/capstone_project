# P300-LLM Speller: EEG + Language-Model Bayesian Fusion

This repository implements an end-to-end P300 brain-computer-interface (BCI) spelling pipeline that combines real EEG evidence with a lightweight language-model prior. The project is centered on **bigP3BCI Study D**, whose recordings use an extended **9×8 keyboard layout with 72 possible classes** rather than the smaller 6×6 matrix used by many classic P300 speller examples.

The main research question is:

> Can a language model improve P300 spelling efficiency by guiding character selection when EEG evidence is uncertain, without overriding confident EEG decisions?

The system answers this by preprocessing raw EDF recordings, training a memory-efficient P300 classifier, replaying character-level trials, and comparing EEG-only decoding against fixed, adaptive, and retrieval-augmented EEG+LLM fusion. The reported RAG condition uses a leakage-safe global pooled phrase bank while preserving the EEG classifier and Bayesian decoder.

---

## 1. Novelty and contribution

### 1.1 Full-grid Study D support instead of a hardcoded 6×6 matrix

A major novelty of this implementation is that the spelling matrix is built from the **actual character channels in the EDF files**. Study D contains a 9×8 extended keyboard with letters, punctuation, and special keys. The pipeline therefore avoids assuming 36 classes and instead loads `n_rows`, `n_cols`, and the `grid_map` produced during preprocessing.

Why this matters:

- A wrong class count changes the decoder posterior dimension.
- A wrong class count changes maximum entropy, which affects adaptive fusion calibration.
- A wrong row/column mapping assigns EEG evidence to the wrong character.
- Special entries such as `Sp` must map to their semantic meaning during LLM prediction.

Implemented in:

- `src/data/batch_preprocess.py` for channel-derived grid extraction.
- `run_pipeline.py` for loading the real grid from `data/processed/grid_layout.json`.
- `src/models/fusion.py` for entropy normalization using the actual number of classes.

### 1.2 LLM prior as a grid-level character distribution

The project does not ask the language model to directly output one character through generation. Instead, it converts a causal language model's next-token distribution into a probability distribution over the P300 grid.

How it works:

1. Split the typed context into completed prior text and the current partial word.
2. Run `distilgpt2` once on the prior text.
3. Decode and sort the tokenizer vocabulary one time.
4. For each candidate grid character, build `partial_word + candidate`.
5. Sum probability mass from all vocabulary tokens whose decoded text starts with that prefix.
6. Normalize the resulting scores across the P300 grid.

Why this is novel for the project:

- It avoids one model call per candidate character.
- It handles subword tokenization instead of assuming one token equals one character.
- It works with incomplete words, where the language model cannot know whether the user intends to stop or continue.
- It produces a distribution that can be fused directly with EEG posterior probabilities.

### 1.3 Bayesian evidence fusion in the log domain

The fusion layer combines EEG evidence and language priors using a Bayesian-style product of experts:

```text
log posterior(character) = log EEG(character) + alpha * log LM(character)
```

The implementation normalizes with `logsumexp` to avoid numerical underflow. It also treats unmapped or unsupported grid entries carefully so zero-probability LLM entries do not corrupt the posterior.

Two fusion modes are supported:

- **Fixed fusion:** constant LLM weight.
- **Adaptive fusion:** LLM influence changes with uncertainty.

### 1.4 Adaptive-file ground truth from `SelectedTarget` pulses

Study D contains adaptive dynamic-stopping files. These files do not always contain a fixed number of sequences per character, so fixed-block segmentation can produce incorrect ground truth.

The project resolves this by reading finalized selections from the `SelectedTarget` channel. This is important for repeated letters: a method based only on target-code transitions can collapse repeated characters such as the two `T`s in `BUTTON`. `SelectedTarget` pulses preserve each finalized selection, including consecutive repeated characters.

### 1.5 Metadata-driven replay of real EEG flashes

The pipeline preserves `StimulusCode` as epoch metadata. This is critical because `epochs.events[:, 2]` stores only the binary target/non-target label; it does not identify which row or column flashed. The evaluation pipeline reconstructs row and column likelihoods from the preserved `stimulus_code` metadata.

For adaptive files, the preprocessing stage also stores:

- `char_index`
- `sequence_in_char`

The replay loop uses these fields to select the correct flash epochs for each character and sequence.


### 1.6 Retrieval-augmented global language prior

The implemented **retrieval-augmented generation (RAG) layer** sits between the typed context and the LLM prior. Instead of relying only on the pretrained `distilgpt2` distribution, it retrieves prefix-compatible entries from a local global phrase bank and uses their continuations to condition the next-character prior.

The reported bank is built from training-split Study-D targets only. Retrieval is prefix-compatible and produces a grid-level next-character prior that is conservatively blended with the base LLM. This supports auditable, low-latency retrieval without a vector database or an oracle target vocabulary.

The predictor also supports subject-aware retrieval for future deployment. It is guarded by retrieval-confidence and data-sufficiency gates, so sparse personal banks defer to the global corpus rather than degrade decoding.

---

## 2. Repository layout

```text
p300-llm-speller/
├── README.md
├── requirements.txt
├── setup_env.sh
├── check_project_status.sh
├── run_pipeline.py
├── plot_threshold_sweep.py
├── test_sgd.py
│
├── data/
│   ├── rag/
│   │   ├── phrase_bank_global.csv   # Train-split global retrieval bank
│   │   ├── by_subject/              # Train-split subject banks
│   │   └── synthetic_validation/    # Isolated gate-validation banks
│   └── raw/                         # Raw dataset location; large files are not committed
│
├── notebooks/                       # Notebook workspace placeholder
├── paper/                           # Report/paper workspace placeholder
│
├── results/
│   ├── tables/
│   │   ├── personalization_ablation_studyd.json
│   │   └── sufficiency_gate_pooled_sweep.json
│   └── bigp3bci_studyd_e_findings/
│       ├── character_trials.csv
│       ├── file_summary.csv
│       ├── grid_layouts.csv
│       ├── quality_flags.csv
│       └── stimulus_code_counts.csv
│
├── scripts/
│   ├── analyze_bigp3bci_studyd_e.py
│   ├── build_phrase_bank_from_registry.py
│   ├── run_personalization_ablations.py
│   ├── build_synthetic_validation_bank.py
│   ├── sweep_sufficiency_gate.py
│   └── sweep_sufficiency_gate_pooled.py
│
└── src/
    ├── data/
    │   ├── batch_preprocess.py
    │   ├── load_bigp3bci.py
    │   └── load_bnci2014008.py
    │
    ├── preprocessing/
    │   ├── build_session_sequence.py
    │   └── epoching.py
    │
    ├── models/
    │   ├── classifier.py
    │   ├── decoder.py
    │   ├── fusion.py
    │   ├── llm_predictor.py
    │   └── rag_predictor.py
    │
    ├── evaluation/
    │   ├── ablations.py
    │   └── metrics.py
    │
    └── utils/
```

---

## 3. Component-by-component documentation

## 3.1 `run_pipeline.py` — end-to-end evaluation driver

### What it does

`run_pipeline.py` orchestrates the full experiment:

1. Loads the Study D spelling matrix from processed grid metadata.
2. Initializes the P300 decoder.
3. Initializes the LLM character-prior predictor.
4. Loads the trained P300 classifier.
5. Replays character trials from the ground-truth registry.
6. Runs five evaluation arms:
   - EEG-only baseline.
   - Fixed-weight EEG+LLM fusion.
   - Adaptive entropy-based EEG+LLM fusion.
   - Fixed RAG-LM fusion.
   - Gated RAG-LM fusion.
7. Reports accuracy, flashes per character, ITR, and approximate WPM.

### Why it exists

The script provides a single executable entry point for comparing decoding strategies under the same dataset, classifier, trial order, and stopping rule.

### How it works

- `load_spelling_matrix()` reads `data/processed/grid_layout.json` and builds a matrix whose dimensions match the real Study D layout.
- `real_eeg_classifier_stream()` loads epoched FIF files lazily, caches them, selects the correct character/sequence epochs, runs the classifier, and maps row/column flash probabilities into a full grid posterior.
- `run_evaluation()` loops through character trials, accumulates EEG evidence, applies the selected predictor prior as an initial bias, and stops when confidence crosses the configured threshold.
- `SimpleAblationTracker` stores and prints summary metrics for each experimental condition.

---

## 3.2 `src/data/batch_preprocess.py` — raw EDF to epoched FIF preprocessing

### What it does

This module converts raw bigP3BCI Study D EDF files into MNE epoch files. It also writes project metadata needed for replay:

- `data/processed/StudyD/*-epo.fif`
- `data/processed/grid_layout.json`
- `data/processed/ground_truth_registry.json`
- `data/processed/quality_flags.csv`
- `data/processed/batch_preprocess_failures.log`

### Why it exists

Raw EDF files contain EEG channels, trigger/state channels, and character-layout channels. The decoder needs clean EEG epochs plus metadata describing which row or column flashed and which target character each trial belongs to.

### How it works

1. Loads one EDF file with MNE.
2. Detects EEG channels by the `EEG_` prefix.
3. Detects grid channels by parsing names like `A_1_1`, including punctuation labels.
4. Renames EEG channels to standard montage-compatible names.
5. Applies a 0.1–30 Hz bandpass filter.
6. Optionally applies a notch filter if configured.
7. Finds stimulus onsets from rising edges in `StimulusBegin`.
8. Reads binary target labels from `StimulusType`.
9. Reads row/column flash identity from `StimulusCode`.
10. Builds MNE epochs from -100 ms to 800 ms around each flash.
11. Stores `stimulus_code` and `stimulus_type` as epoch metadata.
12. For adaptive files, detects selected characters from `SelectedTarget` pulses and stores `char_index` and `sequence_in_char`.
13. Saves epochs and updates JSON registries.

### Fixed vs adaptive segmentation

Fixed calibration recordings use a known number of target flashes per character. Adaptive recordings use variable stopping, so this module uses `SelectedTarget` pulses instead of fixed blocks.

---

## 3.3 `src/preprocessing/epoching.py` — single-file parser and development reference

### What it does

This file contains a direct parser for a single bigP3BCI EDF file. It performs channel parsing, filtering, event extraction, ground-truth extraction, and epoch creation.

### Why it exists

It acts as a focused development/debugging version of the preprocessing logic. It documents the assumptions behind event extraction and fixed-block segmentation.

### How it works

- Reads one EDF file.
- Detects EEG and grid channels.
- Extracts `StimulusBegin`, `StimulusType`, `StimulusCode`, and `CurrentTarget`.
- Creates MNE epochs.
- Preserves `stimulus_code` metadata for downstream row/column reconstruction.

### Important limitation

The module warns that fixed-block segmentation is inappropriate for adaptive `Dyn`/`DynBigram` files. The batch preprocessor contains the newer adaptive `SelectedTarget` logic.

---

## 3.4 `src/preprocessing/build_session_sequence.py` — character-trial generator

### What it does

This module yields one dictionary per character trial. Each yielded trial includes:

- `target_char`
- `context_so_far`
- `eeg_data_path`

### Why it exists

The language model needs the text context preceding the current character. The evaluator also needs to know which processed EEG file contains the flashes for that character.

### How it works

- Reads `data/processed/ground_truth_registry.json` when available.
- Reconstructs the corresponding FIF filename for each session.
- Prefers files under `data/processed/StudyD`.
- Yields each character in each ground-truth string with its preceding context.
- Falls back to mock words if the registry does not exist, allowing pipeline smoke tests before preprocessing is complete.

---

## 3.5 `src/models/classifier.py` — memory-efficient P300 classifier training

### What it does

This module trains a probabilistic EEG classifier over epoched P300 data. It uses an `SGDClassifier` with logistic loss as an incremental, memory-conscious alternative to loading the entire corpus at once.

### Why it exists

The dataset can be too large to train comfortably in memory if all epochs are loaded simultaneously. Incremental training allows file-by-file processing.

### How it works

Training uses two passes:

1. **Scaler pass:** fit a `StandardScaler` incrementally over vectorized EEG epochs.
2. **Classifier pass:** train the logistic classifier with `partial_fit()` over multiple epochs.

The classifier pipeline includes:

- MNE `Scaler`
- MNE `Vectorizer`
- scikit-learn `StandardScaler`
- scikit-learn `SGDClassifier(loss='log_loss')`

The final model is saved to:

```text
data/processed/swlda_model.pkl
```

Although the file name refers to SWLDA-style P300 classification, the current implementation uses an incremental logistic SGD classifier with class weighting for noisy and imbalanced P300 targets.

---

## 3.6 `src/models/decoder.py` — evidence accumulation and ITR

### What it does

The decoder accumulates per-character posterior evidence across flash sequences and returns the most likely grid character plus confidence.

### Why it exists

P300 spellers typically require multiple flash sequences per character. A single sequence may be noisy; accumulating evidence improves confidence.

### How it works

- Initializes one log-probability accumulator per grid cell.
- Adds clipped log probabilities after each EEG sequence.
- Converts accumulated log probabilities back to a normalized posterior with the log-sum-exp trick.
- Returns the highest-probability character and confidence.
- Provides `calculate_itr()` using the standard Wolpaw information-transfer-rate formula.

---

## 3.7 `src/models/fusion.py` — Bayesian fusion engine

### What it does

This module combines EEG posteriors and language-model priors.

### Why it exists

EEG and language context provide complementary information:

- EEG identifies what the user attended to.
- The LLM estimates which character is likely given the typed context.

Fusion can improve efficiency if the LLM helps resolve ambiguous EEG evidence.

### How it works

The fusion engine supports:

- `fixed` mode: use a constant `base_alpha`.
- `adaptive` mode: scale the fusion weight using entropy.
- `toggle()`: disable fusion for EEG-only baseline runs.
- `get_initial_log_bias()`: apply the LLM prior once at the start of a character selection.

Numerical safeguards include:

- Epsilon clipping before logs.
- `logsumexp` normalization.
- Explicit handling of zero-probability unmapped classes.
- Entropy normalization based on the true number of grid classes.

---

## 3.8 `src/models/llm_predictor.py` — language-model character prior

### What it does

This module converts `distilgpt2` next-token probabilities into a probability distribution over the P300 grid.

### Why it exists

The decoder needs an LLM prior over the same classes as the EEG posterior. A language model naturally predicts tokens, not P300 grid cells, so this adapter maps token probabilities to character probabilities.

### How it works

1. Flattens the grid into `char_list`.
2. Loads `AutoTokenizer` and `AutoModelForCausalLM`.
3. Builds a sorted decoded-token index for fast prefix matching.
4. Splits text context into prior text and partial word.
5. Computes one next-token distribution.
6. Sums token probability mass for vocabulary entries matching each candidate prefix.
7. Returns a normalized grid-level probability vector.

Special behavior:

- Empty context returns a uniform prior.
- `Sp` is mapped to an actual space character.
- Multi-character special keys are skipped unless explicitly mapped.
- If no prefix matches produce mass, the predictor falls back to a uniform prior.


---

## 3.9 `src/models/rag_predictor.py` — retrieval-augmented character prior

### What it does

This module wraps an existing base predictor such as `LLMPredictor` and adds a local phrase-bank retrieval prior. It converts prefix-compatible phrase continuations into a next-character probability vector over the same P300 grid classes.

### Why it exists

A pretrained LLM may under-prioritize user-specific names, clinical phrases, commands, or study-specific vocabulary. The RAG predictor adds an explicit memory layer without retraining the EEG classifier or the language model.

### How it works

1. Loads phrase-bank rows from CSV or JSON.
2. Normalizes phrase text and extracts the current partial word from `context_so_far`.
3. Finds phrase-bank entries containing tokens that start with the partial word.
4. Converts the next character after the partial prefix into a grid index.
5. Builds a retrieval-only probability vector from weighted matching phrases.
6. Applies a confidence-aware interpolation weight and blends retrieval with the base LLM prior.
7. Stores diagnostics including matches, confidence, effective global/subject weights, the data-sufficiency value, and the reason.
8. When a subject is supplied, loads `data/rag/by_subject/phrase_bank_<subject>.csv`; subject retrieval falls back to character bigrams when a word match is unavailable.
9. Gates personal evidence by both retrieval confidence and corpus sufficiency. The latter is a sigmoid over subject-bank token count (midpoint: 150 tokens), preventing a small bank from dominating because of one confident match.

### Current usage

`run_pipeline.py` evaluates global RAG alongside EEG-only, fixed-LLM, and adaptive-LLM conditions. `scripts/run_personalization_ablations.py` separately evaluates global, subject-only, and gated subject/global retrieval by held-out subject. Subject-aware retrieval is not a primary Study-D result because the available subject corpora are too small.

---

## 3.10 `scripts/run_personalization_ablations.py` — subject-aware RAG diagnostic

### What it does

Runs a five-rung, held-out Study-D comparison for each subject:

1. Classifier only.
2. Classifier + plain LLM fusion.
3. Classifier + global pooled RAG.
4. Classifier + hard subject-only RAG.
5. Classifier + gated subject/global RAG with within-session memory growth.

It writes per-subject metrics and paired Wilcoxon tests with Holm–Bonferroni adjustment to `results/tables/personalization_ablation_studyd.json`.

### Interpretation

The subject-only and personalized rungs are diagnostic stress tests, not headline performance claims. Their purpose is to verify that the data-sufficiency gate limits the influence of sparse personal corpora and to document the failure mode of naive subject-only retrieval on Study D.

---

## 3.11 `src/evaluation/metrics.py` — metrics and ablation summaries

### What it does

Defines `SpellerMetrics` and an `AblationTracker` utility for evaluation reporting.

### Why it exists

The project compares multiple experimental conditions, so results need consistent metric definitions.

### How it works

Metrics include:

- Total characters.
- Correct characters.
- Total flashes used.
- Total elapsed time.
- Accuracy.
- Average flashes per character.
- Time per character.
- Approximate WPM.

---

## 3.12 `scripts/analyze_bigp3bci_studyd_e.py` — dataset analytics

### What it does

This script scans bigP3BCI Study D and Study E EDF files and writes reusable CSV analytics.

### Why it exists

Before building the final pipeline, the dataset structure needed to be inspected and validated. The script documents file-level properties, grid layouts, stimulus-code counts, character trials, and quality flags.

### How it works

- Recursively finds EDF files.
- Parses path context such as study, subject, session, phase, and condition.
- Reads event/state channels.
- Extracts grid layout information.
- Writes CSV summaries to `results/bigp3bci_studyd_e_findings/`.

---

## 4. End-to-end implementation steps

### Step 1 — Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core dependencies include MNE, NumPy, pandas, SciPy, scikit-learn, PyTorch, and Hugging Face Transformers.

### Step 2 — Place raw data

Place the bigP3BCI dataset under:

```text
data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyD/
```

The repository intentionally does not commit raw EDF files or generated FIF files because they are large data artifacts.

### Step 3 — Inspect dataset structure

Optional but recommended:

```bash
python scripts/analyze_bigp3bci_studyd_e.py \
  --data-root data/raw/bigP3BCI_dataset \
  --output-dir results/bigp3bci_studyd_e_findings
```

This creates CSV summaries that help verify grid size, stimulus codes, and possible quality issues.

### Step 4 — Preprocess Study D EDF files

```bash
python src/data/batch_preprocess.py
```

Expected generated files:

```text
data/processed/StudyD/*-epo.fif
data/processed/grid_layout.json
data/processed/ground_truth_registry.json
data/processed/quality_flags.csv
```

During this step, the pipeline:

- Extracts EEG channels.
- Extracts the real 9×8 grid.
- Filters EEG data.
- Creates epochs around stimulus onsets.
- Preserves stimulus metadata.
- Detects adaptive character boundaries with `SelectedTarget` pulses.

### Step 5 — Train the classifier

```bash
python src/models/classifier.py
```

Expected output:

```text
data/processed/swlda_model.pkl
```

This trains a probabilistic P300 classifier using file-by-file incremental learning.

### Step 6 — Build leakage-safe retrieval banks

```bash
python scripts/build_phrase_bank_from_registry.py
```

This deterministically splits sessions per subject, writes the global and per-subject training banks, and records the held-out session IDs. Run it after preprocessing whenever the ground-truth registry changes.

### Step 7 — Run the full evaluation

```bash
python run_pipeline.py
```

This evaluates:

1. EEG-only baseline.
2. Fixed LLM fusion.
3. Adaptive LLM fusion.
4. Fixed RAG-LM fusion.
5. Gated RAG-LM fusion.

The script prints an ablation summary with:

- Accuracy.
- Flashes per character.
- ITR.
- WPM approximation.

### Step 8 — Run the subject-aware diagnostic (optional)

```bash
python scripts/run_personalization_ablations.py
```

This takes longer than the main pipeline because it evaluates all five rungs for every held-out subject. It writes the per-subject results and corrected paired tests to `results/tables/personalization_ablation_studyd.json`.

### Step 9 — Validate the sufficiency gate with isolated synthetic banks (optional)

```bash
# Requires a local public-domain filler text at
# data/rag/synthetic_validation/filler_source.txt
python scripts/build_synthetic_validation_bank.py \
  --subject D_07 --sizes 18 50 150 500 1000
python scripts/sweep_sufficiency_gate.py \
  --subject D_07 --sizes 18 50 150 500 1000
```

Synthetic banks are tagged `synthetic_validation`, exclude the selected subject's held-out target words, and remain outside the real per-subject bank directory. They validate gate behavior only; do not combine them with the Study-D ablation results.

For a pooled, raw-count summary after generating banks for each selected subject:

```bash
python scripts/sweep_sufficiency_gate_pooled.py
```

### Step 10 — Plot threshold sweep if needed

```bash
python plot_threshold_sweep.py
```

This produces or updates:

```text
results/threshold_sweep.png
```

---

## 5. Major changes completed in the project

### 5.1 Replaced hardcoded 6×6 decoding assumptions

Earlier P300 examples often assume a 6×6 matrix. This project now builds the matrix from the actual EDF-derived grid layout and propagates the real class count through the decoder, fusion layer, and evaluation pipeline.

### 5.2 Preserved `StimulusCode` metadata

The preprocessing pipeline now stores `stimulus_code` per epoch. This enables the evaluator to know which row or column flashed when reconstructing grid-level posteriors.

### 5.3 Added adaptive segmentation metadata

Adaptive files now include per-epoch `char_index` and `sequence_in_char`. This allows replay of variable-length character selections without assuming a fixed number of flash blocks.

### 5.4 Fixed adaptive ground-truth extraction

Ground truth for dynamic files now comes from `SelectedTarget` pulses rather than fragile target-code transitions or fixed blocks. This preserves repeated characters and prevents merged-letter errors.

### 5.5 Implemented memory-efficient classifier training

The classifier trainer uses incremental scaling and `partial_fit()` to avoid loading the complete dataset into memory.

### 5.6 Implemented prefix-marginalized LLM priors

The language predictor now sums probability mass across all decoded vocabulary tokens matching each candidate prefix, rather than relying on a single generated token or repeated model calls.

### 5.7 Added fixed and adaptive Bayesian fusion

The project now supports both a constant-weight fusion condition and an entropy-aware adaptive fusion condition.

### 5.8 Added real-data replay with fallback safety

The evaluator loads real FIF epochs when available and falls back to mock data only when required for smoke testing. This allows development even before all large data artifacts are present.

### 5.9 Added dataset analytics outputs

The `results/bigp3bci_studyd_e_findings/` CSV files provide auditable summaries of Study D/E file structure, grid layout, stimulus counts, inferred character trials, and quality flags.

### 5.10 Implemented global and subject-aware RAG priors

The project now includes `RAGPredictor`, a local phrase-bank retrieval layer that wraps the base LLM predictor and converts prefix-compatible phrase continuations into grid-level next-character probabilities. Global and optional subject banks are blended with separate continuous confidence gates. Personal evidence also passes a token-count sufficiency gate, with character-bigram backoff when word retrieval finds no match.

### 5.11 Added held-out personalization and gate-validation experiments

The repository now includes a per-subject five-rung RAG ablation with paired, Holm–Bonferroni-corrected Wilcoxon comparisons. A separate synthetic-corpus sweep validates the data-sufficiency mechanism without entering the real Study-D phrase-bank or headline ablation results.

---

## 6. Experimental design

The primary experiment is a five-arm ablation:

| Condition | EEG evidence | LLM prior | Fusion behavior |
|---|---|---|---|
| Baseline | Yes | No | Fusion disabled |
| Fixed fusion | Yes | Yes | Constant alpha |
| Adaptive fusion | Yes | Yes | Entropy-scaled alpha |
| Fixed RAG-LM fusion | Yes | RAG-enhanced | Constant alpha |
| Gated RAG-LM fusion | Yes | RAG-enhanced | Entropy-scaled alpha plus retrieval confidence gating |

The decoding loop applies the selected LLM or RAG-LM prior as an initial log bias for each character. EEG evidence is then accumulated sequence by sequence until either:

- confidence exceeds the threshold, or
- the maximum number of sequences is reached.

This design measures whether language context can reduce the number of flashes required while preserving or improving accuracy.

### 6.1 Main five-arm results (held-out test split, real EEG + SWLDA)

The RAG phrase bank is built exclusively from an 80% train split of Study D sessions (`scripts/build_phrase_bank_from_registry.py`); evaluation runs only on the remaining 20% held-out sessions (`data/processed/test_sessions.json`), so no target-vocabulary leaks between the phrase bank and the reported numbers. `rag_weight`/`retrieval_confidence_threshold` were selected via a 15-point grid sweep (`scripts/sweep_rag_params.py`, results in `results/tables/rag_sweep.json`) on that same train split.

**Reported RAG condition.** The primary RAG result uses the global pooled
phrase bank (rung 2). This is the competitive, non-regressing condition for
Study D; it is not a claim that the fixed Study-D corpus supports reliable
subject-specific personalization.

**Personalization safety mechanism.** `RAGPredictor` can automatically blend
a subject corpus with the global corpus when a subject identity is available.
Its subject contribution is governed by two independent continuous gates:
retrieval confidence and data sufficiency (a sigmoid of the subject-bank token
count). The latter prevents a small corpus from being trusted merely because a
single retrieval match is confident. Subject-only and blended-personalization
rungs are retained as diagnostic ablations, not as the headline Study-D RAG
claim. In Study D, each subject has at most 18 distinct words even after
pooling the available conditions, which is below the evidence needed to trust
a personal corpus. Naive subject-only retrieval therefore regresses versus
plain LM/global RAG; this failure motivates the sufficiency gate, which defers
to the global corpus for this dataset.

The synthetic sufficiency-gate sweep is a separately labelled
mechanism-validation experiment. Its generated banks use
`source="synthetic_validation"`, are stored outside `data/rag/by_subject/`,
and are never included in the held-out Study-D results.

| Condition | Accuracy (%) | Flashes/Char | ITR (bits/min) |
|---|---|---|---|
| Baseline (No LLM) | 32.66 | 12.82 | 2.62 |
| Fusion (Fixed a=0.1) | 53.30 | 9.62 | 7.18 |
| Fusion (Adaptive) | 52.15 | 9.68 | 6.91 |
| **Fusion (Fixed RAG-LM)** | **54.15** | **9.59** | **7.37** |
| Fusion (Gated RAG-LM) | 52.72 | 9.68 | 7.01 |

Full table: `results/tables/ablation_results.csv`. Figures: `results/figures/ablation_comparison.png` (per-metric bars), `results/figures/accuracy_vs_itr.png` (tradeoff scatter).

**Takeaway:** any LLM prior roughly triples ITR over EEG-only decoding (2.62 → ~7). The reported global RAG condition is competitive with plain fixed fusion while using a leakage-safe, session-held-out pooled phrase bank rather than an oracle one. Per-subject retrieval is an architectural capability with an explicit data-sufficiency safeguard, not a performance claim for Study D's fixed-vocabulary corpus.

### 6.2 Subject-aware RAG diagnostic

`results/tables/personalization_ablation_studyd.json` records a separate five-rung analysis over 17 held-out subjects. It compares classifier-only, plain LLM, global RAG, hard subject-only RAG, and the gated subject/global configuration. The comparisons are paired at the subject level and use Holm–Bonferroni correction across the planned tests.

The diagnostic confirms the intended limitation: hard subject-only retrieval is materially worse than global RAG on this sparse fixed-vocabulary corpus. The gated personalized rung likewise should not be read as a gain claim for Study D. It demonstrates the fallback architecture; global pooled RAG remains the supported retrieval condition for this dataset.

### 6.3 Synthetic sufficiency-gate validation

The synthetic sweep varies only the size of isolated, non-target-containing phrase banks while retaining real held-out EEG sessions. At the configured 150-token midpoint, the sufficiency gate is 0.5; it is near zero for 18/50-token banks and approaches one for 500/1000-token banks. Results are stored in `results/tables/sufficiency_gate_sweep.json` and `results/tables/sufficiency_gate_pooled_sweep.json` and are mechanism-validation outputs, not a Study-D personalization benchmark.

---

## 7. Key assumptions and caveats

1. **Raw and processed datasets are external artifacts.** The repository expects them under `data/raw` and `data/processed`, but they are not committed.
2. **Study D layout is expected to be stable.** The batch preprocessor raises an error if a later file disagrees with the first saved Study D grid layout.
3. **The LLM/RAG prior is only as good as the text context and phrase bank.** If the ground-truth registry is incomplete, malformed, or poorly aligned with the RAG phrase bank, contextual predictions degrade.
4. **The classifier must match the processed epoch format.** If old FIF files lack required metadata, they should be regenerated.
5. **Adaptive fusion depends on the correct number of classes.** The entropy denominator must match the true grid size.
6. **Timing in ITR is approximated from flashes.** The current evaluator estimates average time per character using flashes and a fixed timing approximation.
7. **Personalized retrieval needs enough prior text.** Study D's per-subject banks are sparse; the global bank is the reported condition. The subject path is deliberately gated and should be re-evaluated with a richer deployment corpus.
8. **Synthetic gate-validation banks are not training data.** Keep generated files in `data/rag/synthetic_validation/` and do not merge them into `data/rag/by_subject/`.

---

## 8. Troubleshooting

### `grid_layout.json` is missing

Run:

```bash
python src/data/batch_preprocess.py
```

### `swlda_model.pkl` is missing

Run:

```bash
python src/models/classifier.py
```

### The pipeline warns about legacy adaptive indexing

Regenerate processed FIF files with the current batch preprocessor so adaptive metadata columns are available.

### The pipeline falls back to mock data

Check that:

- `data/processed/ground_truth_registry.json` exists.
- FIF files exist under `data/processed/StudyD/`.
- Registry keys match processed FIF stems.

### Hugging Face model download fails

Ensure network access is available or pre-cache `distilgpt2` in the execution environment.

### Subject-aware scripts cannot find a phrase bank or split

Run `python scripts/build_phrase_bank_from_registry.py` after preprocessing. It creates `data/rag/phrase_bank_global.csv`, per-subject banks, and the held-out session manifests consumed by the diagnostic scripts.

---

## 9. Quick command reference

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze Study D/E EDF structure
python scripts/analyze_bigp3bci_studyd_e.py \
  --data-root data/raw/bigP3BCI_dataset \
  --output-dir results/bigp3bci_studyd_e_findings

# Preprocess Study D raw EDF files
python src/data/batch_preprocess.py

# Train classifier
python src/models/classifier.py

# Run ablation evaluation
python run_pipeline.py

# Rebuild leakage-safe global and per-subject phrase banks
python scripts/build_phrase_bank_from_registry.py

# Run per-subject personalization diagnostic
python scripts/run_personalization_ablations.py

# Run a one-subject synthetic sufficiency-gate check
python scripts/build_synthetic_validation_bank.py --subject D_07 --sizes 18 50 150 500 1000
python scripts/sweep_sufficiency_gate.py --subject D_07 --sizes 18 50 150 500 1000

# Generate threshold plot
python plot_threshold_sweep.py
```


---

## 10. RAG layer implementation

The implemented RAG path is local, auditable, and uses no external vector database:

```text
context -> global phrase-bank prefix retrieval -> grid-level retrieval prior
        -> confidence-aware blend with LLM prior -> Bayesian EEG fusion -> decoder
```

`scripts/build_phrase_bank_from_registry.py` creates the leakage-safe global
bank from training sessions and writes the held-out evaluation split. Each bank
row contains `text`, `source`, `weight`, and `category`.

`src/models/rag_predictor.py` normalizes phrase text, retrieves compatible
continuations, maps them to actual grid characters, and records diagnostics.
The retrieval influence is continuous and bounded so the base LLM retains
influence. RAG never replaces EEG evidence, the SWLDA classifier, or the
decoder.

Subject-aware retrieval is intentionally not a reported Study-D outcome. If a
future deployment supplies a sufficiently large subject corpus, the same
predictor can load it, use character-bigram backoff for missing word matches,
and apply confidence and data-sufficiency gates before adding any personal
signal. With Study D's limited per-subject vocabulary, the gate defers to the
global corpus.

---

## 11. Project status

Implemented:

- EDF parsing.
- Study D grid extraction.
- P300 epoching.
- Fixed and adaptive ground-truth handling.
- Stimulus metadata preservation.
- Incremental classifier training.
- Character-level decoding.
- LLM character-prior prediction.
- RAG-enhanced character-prior prediction.
- Subject-aware retrieval with confidence, data-sufficiency, and character-bigram backoff safeguards.
- Bayesian fusion.
- Five-arm ablation evaluation.
- Dataset analytics exports.
- Train/test session split with leakage-safe RAG phrase bank (`scripts/build_phrase_bank_from_registry.py`).
- RAG hyperparameter sweep over `rag_weight` and `retrieval_confidence_threshold` (`scripts/sweep_rag_params.py`).
- Final five-arm results measured on real EEG + SWLDA over the held-out test split (§6.1).
- Per-subject five-rung diagnostic with corrected paired comparisons (`scripts/run_personalization_ablations.py`).
- Isolated synthetic sufficiency-gate validation and pooled raw-count summary.

Planned next work:

- Aggregate RAG diagnostics (retrieval enabled-rate, match reasons) in the evaluation summary.
- Populate `results/figures/` and `paper/` with the full writeup for arXiv/workshop submission.
- Replicate the personalization diagnostic across seeds and richer subject corpora.
- Evaluate retrieval as an isolated component (LM-only versus LM + retrieval) beyond the combined fusion arms.
- Expand the phrase bank with consented, deployment-representative data once available.
