# P300-LLM Speller: EEG + Language-Model Bayesian Fusion

This repository implements an end-to-end P300 brain-computer-interface (BCI) spelling pipeline that combines real EEG evidence with a lightweight language-model prior. The project is centered on **bigP3BCI Study D**, whose recordings use an extended **9×8 keyboard layout with 72 possible classes** rather than the smaller 6×6 matrix used by many classic P300 speller examples.

The main research question is:

> Can a language model improve P300 spelling efficiency by guiding character selection when EEG evidence is uncertain, without overriding confident EEG decisions?

The system answers this by preprocessing raw EDF recordings, training a memory-efficient P300 classifier, replaying character-level trials, and comparing EEG-only decoding against fixed and adaptive EEG+LLM fusion.

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

---

## 2. Repository layout

```text
capstone_project/
├── README.md
├── requirements.txt
├── setup_env.sh
├── check_project_status.sh
├── run_pipeline.py
├── plot_threshold_sweep.py
├── test_sgd.py
│
├── data/
│   └── raw/                         # Raw dataset location; large files are not committed
│
├── notebooks/                       # Notebook workspace placeholder
├── paper/                           # Report/paper workspace placeholder
│
├── results/
│   ├── threshold_sweep.png
│   └── bigp3bci_studyd_e_findings/
│       ├── character_trials.csv
│       ├── file_summary.csv
│       ├── grid_layouts.csv
│       ├── quality_flags.csv
│       └── stimulus_code_counts.csv
│
├── scripts/
│   └── analyze_bigp3bci_studyd_e.py
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
    │   └── llm_predictor.py
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
6. Runs three evaluation arms:
   - EEG-only baseline.
   - Fixed-weight EEG+LLM fusion.
   - Adaptive entropy-based EEG+LLM fusion.
7. Reports accuracy, flashes per character, ITR, and approximate WPM.

### Why it exists

The script provides a single executable entry point for comparing decoding strategies under the same dataset, classifier, trial order, and stopping rule.

### How it works

- `load_spelling_matrix()` reads `data/processed/grid_layout.json` and builds a matrix whose dimensions match the real Study D layout.
- `real_eeg_classifier_stream()` loads epoched FIF files lazily, caches them, selects the correct character/sequence epochs, runs the classifier, and maps row/column flash probabilities into a full grid posterior.
- `run_evaluation()` loops through character trials, accumulates EEG evidence, applies the LLM prior as an initial bias, and stops when confidence crosses the configured threshold.
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

## 3.9 `src/evaluation/metrics.py` — metrics and ablation summaries

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

## 3.10 `scripts/analyze_bigp3bci_studyd_e.py` — dataset analytics

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

### Step 6 — Run the full evaluation

```bash
python run_pipeline.py
```

This evaluates:

1. EEG-only baseline.
2. Fixed LLM fusion.
3. Adaptive LLM fusion.

The script prints an ablation summary with:

- Accuracy.
- Flashes per character.
- ITR.
- WPM approximation.

### Step 7 — Plot threshold sweep if needed

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

---

## 6. Experimental design

The primary experiment is a three-arm ablation:

| Condition | EEG evidence | LLM prior | Fusion behavior |
|---|---|---|---|
| Baseline | Yes | No | Fusion disabled |
| Fixed fusion | Yes | Yes | Constant alpha |
| Adaptive fusion | Yes | Yes | Entropy-scaled alpha |

The decoding loop applies the LLM prior as an initial log bias for each character. EEG evidence is then accumulated sequence by sequence until either:

- confidence exceeds the threshold, or
- the maximum number of sequences is reached.

This design measures whether language context can reduce the number of flashes required while preserving or improving accuracy.

---

## 7. Key assumptions and caveats

1. **Raw and processed datasets are external artifacts.** The repository expects them under `data/raw` and `data/processed`, but they are not committed.
2. **Study D layout is expected to be stable.** The batch preprocessor raises an error if a later file disagrees with the first saved Study D grid layout.
3. **The LLM prior is only as good as the text context.** If the ground-truth registry is incomplete or malformed, contextual predictions degrade.
4. **The classifier must match the processed epoch format.** If old FIF files lack required metadata, they should be regenerated.
5. **Adaptive fusion depends on the correct number of classes.** The entropy denominator must match the true grid size.
6. **Timing in ITR is approximated from flashes.** The current evaluator estimates average time per character using flashes and a fixed timing approximation.

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

# Generate threshold plot
python plot_threshold_sweep.py
```

---

## 10. Project status

Implemented:

- EDF parsing.
- Study D grid extraction.
- P300 epoching.
- Fixed and adaptive ground-truth handling.
- Stimulus metadata preservation.
- Incremental classifier training.
- Character-level decoding.
- LLM character-prior prediction.
- Bayesian fusion.
- Three-arm ablation evaluation.
- Dataset analytics exports.

The main remaining work is to run the final experiments on the full local data artifacts and replace any placeholder result values with final measured metrics.
