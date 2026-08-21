# P300-LLM Speller

A P300 event-related-potential (ERP) speller that fuses classifier posteriors with a lightweight language model, extending Bayesian dynamic-stopping / language-model work with a modern LLM-based predictive prior.

- **Repo:** [shivankbhatia/capstone_project](https://github.com/shivankbhatia/capstone_project/tree/SKB-J8?utm_source=chatgpt.com)
- **Active branch:** `SKB-J8`
- **Dataset:** bigP3BCI Study D on PhysioNet — 307 EDF files with a 9×8 grid containing 72 character classes.
- **Project objective:** evaluate whether LLM-guided Bayesian fusion improves spelling accuracy, reduces flashes required per character, and improves information transfer rate (ITR).

---

## 1. Project goal

Classic P300 spellers decode the attended grid cell primarily from accumulated EEG evidence. This project combines EEG posterior probabilities with a character-level language prior.

The evaluation compares three decoding strategies:

1. **Baseline** — EEG classifier evidence only
2. **Fixed fusion** — EEG evidence fused with a constant LLM weight, `α = 0.1`
3. **Adaptive fusion** — LLM influence adapts according to decoding uncertainty

The central hypothesis is that a predictive language prior can improve character selection efficiency when EEG evidence is uncertain, while allowing strong EEG evidence to dominate when confidence is high.

---

## 2. Results

The final evaluation uses the same dataset and decoding pipeline across all three experimental arms.

| Method | Accuracy | Avg. Flashes / Character | ITR |
|---|---:|---:|---:|
| **Baseline** | **[INSERT RESULT]** | **[INSERT RESULT]** | **[INSERT RESULT]** |
| **Fixed Fusion (`α = 0.1`)** | **[INSERT RESULT]** | **[INSERT RESULT]** | **[INSERT RESULT]** |
| **Adaptive Fusion** | **[INSERT RESULT]** | **[INSERT RESULT]** | **[INSERT RESULT]** |

### Interpretation

- **Accuracy** measures the proportion of correctly decoded characters.
- **Flashes per character** measures decoding efficiency; fewer flashes indicate faster selections.
- **Information Transfer Rate (ITR)** combines accuracy, number of possible classes, and selection time to measure overall communication throughput.
- The baseline establishes performance using EEG evidence alone.
- Fixed fusion evaluates whether a small constant language-prior contribution (`α = 0.1`) improves decoding.
- Adaptive fusion dynamically adjusts prior influence according to uncertainty, allowing the system to use stronger language guidance when the EEG evidence is less decisive.

> **Note:** Replace the three placeholder rows above with the exact final experimental metrics produced by `run_pipeline.py`.

---

## 3. Repository structure

```text
capstone_project/
├── run_pipeline.py                  # Main end-to-end evaluation pipeline
├── requirements.txt
├── setup_env.sh
│
├── data/
│   ├── raw/                         # gitignored — bigP3BCI EDF files
│   └── processed/                   # gitignored — generated FIF epochs and registries
│
├── src/
│   ├── data/
│   │   ├── batch_preprocess.py      # Full-corpus EDF → epoched FIF processing
│   │   ├── load_bigp3bci.py
│   │   └── load_bnci2014008.py      # Legacy dataset loader
│   │
│   ├── preprocessing/
│   │   ├── epoching.py
│   │   └── build_session_sequence.py
│   │
│   ├── models/
│   │   ├── classifier.py            # Calibrated LDA / SWLDA-style P300 classifier
│   │   ├── decoder.py               # Evidence accumulation and Wolpaw ITR
│   │   ├── fusion.py                # Bayesian fixed/adaptive fusion engine
│   │   └── llm_predictor.py         # distilGPT2 character-prior predictor
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── ablations.py
│       └── baseline_eval.py
│
├── scripts/
│   ├── analyze_bigp3bci_studyd_e.py
│   └── run_pipeline.py
│
└── results/
    ├── bigp3bci_studyd_e_findings/
    └── threshold_sweep.png
```

---

## 4. Core methodology

### EEG decoding

The P300 classifier produces class probabilities from EEG evidence accumulated across flashes. The decoder supports the full Study D keyboard with dynamically configured class counts.

### Fixed Bayesian fusion

The fusion engine combines EEG and language-model probabilities in the log domain:

\[
\log P(c \mid EEG, LM)
=
\log P(c \mid EEG)
+
\alpha \log P(c \mid LM)
\]

For the fixed-fusion experiment:

\[
\alpha = 0.1
\]

The result is normalized using `logsumexp` for numerical stability.

### Adaptive fusion

Adaptive fusion adjusts the influence of the LLM prior based on uncertainty rather than using a fixed contribution.

The fusion engine also exposes an initial log-bias mechanism that scales language-prior influence according to LLM confidence, while preserving numerical stability for zero-probability or unmapped grid entries.

### LLM character prior

`LLMPredictor` uses `distilgpt2` to construct a probability distribution over the actual characters available on the P300 spelling grid.

The predictor:

1. Separates completed context from the current partial word.
2. Performs a single forward pass to obtain the next-token distribution.
3. Builds a searchable index of decoded vocabulary tokens.
4. For each candidate grid character, constructs `partial_word + candidate`.
5. Marginalizes probability mass over all vocabulary tokens whose decoded text begins with that candidate prefix.
6. Normalizes the resulting probabilities across the P300 grid.

This prefix-filtering approach avoids per-candidate model generation and resolves ambiguity around incomplete words and token boundaries.

---

## 5. Pipeline stages

| Stage | Status |
|---|---|
| EDF parsing and channel classification | ✅ Complete |
| Grid-layout extraction | ✅ Complete |
| Fixed-block ground-truth segmentation | ✅ Complete |
| Adaptive ground-truth segmentation | ✅ Complete |
| `StimulusCode` metadata preservation | ✅ Complete |
| Metadata-driven flash indexing | ✅ Complete |
| Session sequence construction | ✅ Complete |
| P300 classifier | ✅ Complete |
| Character decoder | ✅ Complete |
| Fixed Bayesian fusion | ✅ Complete |
| Adaptive Bayesian fusion | ✅ Complete |
| LLM predictive prior | ✅ Complete |
| Three-arm ablation pipeline | ✅ Complete |
| Accuracy / flashes / ITR evaluation | ✅ Complete |
| LLM predictor correctness issues | ✅ Resolved |

---

## 6. LLM predictor correctness resolution

The LLM predictor went through several iterations involving subtle tokenization and probability-assignment issues. These issues are now resolved.

### Resolved issues

- **Casing mismatch:** context and candidate prefixes are normalized consistently.
- **Subword probability marginalization:** probability mass is summed across all vocabulary tokens matching a candidate prefix instead of relying on a single token.
- **Word-boundary ambiguity:** prediction does not depend on determining whether the current partial word is "complete."
- **Generation vs. filtering:** the model performs one next-token prediction from the preceding context, followed by vocabulary prefix filtering, rather than generating independently for every candidate character.
- **Grid mapping:** special grid entries such as `Sp` are mapped correctly to their semantic character representation.
- **Zero-probability handling:** unmapped classes are excluded without allowing numerical edge cases to corrupt adaptive fusion.

The final implementation therefore produces a consistent grid-level language prior suitable for direct Bayesian fusion with EEG posterior probabilities.

---

## 7. Preprocessing and ground-truth construction

### Fixed-block files

For RC/Train calibration files:

```text
sequences_per_selection = 20
```

This fixed-block structure is used for calibration-style recordings.

### Adaptive files

For Dyn/DynBigram recordings, character boundaries are derived directly from pulses in the `SelectedTarget` channel.

This resolves the critical doubled-letter failure mode that occurred when segmentation depended only on transitions in `CurrentTarget`.

The final preprocessing approach correctly handles consecutive identical selections such as:

```text
BUTTON
HIDDEN
ALLOWS
INDEED
```

Adaptive epochs are annotated with metadata including:

- `char_index`
- `sequence_in_char`

The evaluation pipeline uses this metadata for flash indexing rather than assuming fixed-size character blocks.

---

## 8. Environment

- Python 3.9.6
- `mne`
- `moabb`
- `numpy`
- `torch`
- `transformers`
- `scikit-learn`

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --break-system-packages
```

---

## 9. Reproducing the pipeline

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Preprocess the Study D corpus
python src/data/batch_preprocess.py

# 3. Train / run the classifier pipeline
python src/models/classifier.py

# 4. Run the three-arm evaluation
python run_pipeline.py
```

The final evaluation compares:

```text
Baseline
   ↓
Fixed Fusion (α = 0.1)
   ↓
Adaptive Fusion
```

using:

- Character accuracy
- Average flashes per character
- Information Transfer Rate

---

## 10. Known limitations

- Retrieval-augmented or personalized language priors remain a potential future extension.
- Statistical significance testing across experimental conditions can further strengthen the evaluation.
- Legacy files such as `baseline_eval.py` and `scripts/run_pipeline.py` should be consolidated or removed if they are no longer part of the active execution path.

The previously documented **LLM predictor bug chain is no longer an open gap**. Casing normalization, subword marginalization, token-boundary ambiguity, and generation-versus-prefix-filtering issues have been resolved in the current implementation.

---

## 11. References

- Mainsah, B. O., Throckmorton, C. S., et al. — Bayesian dynamic-stopping and language-model approaches for P300 spelling.
- bigP3BCI dataset — PhysioNet.
- distilGPT2 — Hugging Face Transformers.