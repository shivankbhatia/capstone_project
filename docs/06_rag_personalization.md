# Component 6 — Retrieval-augmented prior and personalization (FINAL)

**Status:** FINAL after review against the repository code, locks, tracked banks, and result artifacts on 2026-11-18.

## Purpose and interface

`src/models/rag_predictor.py` implements `RAGPredictor`, a local phrase-bank adapter around the base language-model predictor. It consumes the typed context and the base LM's grid-aligned prior, then returns a modified probability vector over the same grid classes. Phrase banks may be CSV or JSON entries with `text`, optional `source`, `weight`, and `category` fields. `predict_next_char()` returns the grid prior and stores a `RAGDiagnostics` record; `retrieval_prior()` returns the weighted retrieval vector and diagnostics including context, matched phrases, confidence, source, and gate weights.

The implementation supports a shared/global phrase bank, a subject bank, or both. It can add completed decoded text to an in-memory subject bank during a session. It does not persist these updates. Component 5 documents the LM scoring and fusion equations; this chapter documents retrieval and bank effects only.

## RAG algorithm

The steps below follow `RAGPredictor` in `src/models/rag_predictor.py`.

1. **Normalize context and bank entries.** Lowercase text, replace underscores with spaces, and collapse whitespace. Context normalization preserves a trailing space when the supplied context ends in whitespace.
2. **Search context suffixes.** Build context suffixes at word boundaries, longest first. Match each suffix inside each bank phrase only at the phrase start or after a space. For a usable match, take the following character as the continuation; token mode instead looks for the longest grid-key mapping among up to four following whitespace-delimited tokens.
3. **Score matches.** For each phrase, retain its highest-scoring match. Its score is `phrase.weight × (1 + len(matched_context)) × context_bonus`. The bonus is `2` when the complete normalized context matches from the beginning of the phrase, otherwise `1`. Sort by matched-context length and score, then retain at most `max_retrieved_phrases` matches.
4. **Make retrieval distributions.** Add match scores into the corresponding grid classes and normalize each bank's scores. Retrieval confidence is the maximum class probability. Unsupported next characters are discarded; no compatible matches produce a zero retrieval vector and a `no_matches` or `no_grid_compatible_next_char` diagnostic. A missing bank path loads as an empty bank, so retrieval has no influence and the base prior is returned; callers must inspect diagnostics to distinguish this from an active retrieval prior.
5. **Apply confidence gates.** Global confidence is passed through a sigmoid with sharpness `30` and midpoint `retrieval_confidence_threshold`, if the global bank has a usable prior and `subject_only` is false. Subject confidence uses a sigmoid with sharpness `subject_gate_sharpness` and midpoint `subject_confidence_threshold`, if there is a subject word match or an accepted character-bigram backoff. These are continuous gates. Although the constructor docstring calls the retrieval threshold a minimum, the implementation does not make it a binary cutoff.
6. **Apply subject-data sufficiency.** `subject_token_count` is the sum of whitespace-token counts across subject-bank entries. The sufficiency value is `sigmoid(sufficiency_sharpness × (subject_token_count − sufficiency_midpoint_tokens))`. A second subject-bank size factor is `min(1, phrase_count / min_subject_phrases)`. The code comment says the token-count gate is independent of query match confidence and is intended to prevent a sparse bank from gaining influence from one confident-looking match.
7. **Apply gated character-bigram backoff.** When a subject phrase match is absent, a subject bank exists, and backoff is enabled, count character successors from within bank entries. Exclude any successor below `bigram_min_count`; normalize successor entropy by `log(number of supported successors)` and reject the prior if this exceeds `bigram_max_normalized_entropy`. The remaining supported successor counts form a normalized grid prior. Token mode disables this character-level backoff because character bigrams do not preserve multi-character key boundaries.
8. **Blend banks and cap retrieval influence.** Global weight is `rag_weight × global_gate`. Subject weight is `rag_weight × personalization_bonus × subject_phrase_count_weight × subject_confidence_gate × sufficiency_gate`. If their sum exceeds `.9`, scale both proportionally so the sum is `.9`. Combine the weighted global and subject priors, then normalize.
9. **Blend with the base LM prior.** Return the normalized mixture `(1 − total_retrieval_weight) × base_prior + total_retrieval_weight × retrieval_prior`. If retrieval has no weight, the result is the base prior.
10. **Update within a session only.** `add_observed_phrase(text)` and `update(decoded_text)` append a normalized phrase to the in-memory subject bank and update token and bigram counts. They do not write files. The method documentation says callers should supply completed decoded text, not a hidden target. The participant lock requires scoring each next key before a teacher-forced decoded-prefix update; this ordering is a caller protocol, not automatic behavior inside `RAGPredictor`.

The gates have implementation-level rationale in code comments and design fields. The Study D manifest diagnostics show subject contribution is frequently suppressed, and the public-domain development benchmark has degradation cases. Neither establishes that these gates protect real participant performance; no participant data exists.

## Constructor defaults and locked rung settings

| Parameter | `RAGPredictor` default | Locked Study D RAG value / use |
|---|---:|---|
| `phrase_bank_path` | `data/rag/phrase_bank.csv` | Rungs 2–4 use `data/rag/phrase_bank_global.csv`; the subject file is derived under `by_subject/` when `subject_id` is supplied |
| `rag_weight` | `.25` | Rung 2 `.7`; Rung 3 `1.0`; Rung 4 `.7` |
| `retrieval_confidence_threshold` | `.60` | Rung 2 `0.0`; Rung 3 `.6`; Rung 4 `0.0` |
| `max_retrieved_phrases` | `20` | Not overridden in the ladder |
| `min_subject_phrases` | `10` | Lock’s `subject_min_unique_phrases=10` |
| `personalization_bonus` | `1.5` | Lock’s `personalization_bonus=1.5` |
| `subject_only` | `false` | `true` for Rung 3 only |
| `subject_confidence_threshold` | `.20` | Lock `.20` |
| `sufficiency_midpoint_tokens` | `15` | Lock `15.0` tokens |
| `sufficiency_sharpness` | `.20` | Lock `.20` |
| `subject_gate_sharpness` | `10` | Default; not separately overridden by the lock |
| `disable_bigram_backoff` | `false` | Lock enables the configured backoff gate |
| `bigram_min_count` | `2` | Lock `2` |
| `bigram_max_normalized_entropy` | `.65` | Lock `.65` |
| `token_mode` | `false` | Not overridden for the Study D character grid |
| `rung_id`, `debug_trace_calls` | `None`, `0` | Diagnostic labels/tracing; not a locked RAG weight |

For Rung 3, `subject_only=true` disables the global confidence gate. Its configured `.6` retrieval threshold therefore does not gate subject retrieval; the subject-confidence midpoint remains `.20`. The base adaptive fusion multiplier for Rungs 2–4 is handled by Component 5 and is driven by LM prior entropy; it is not a RAG gate.

### Study D ladder interface

| Rung | RAG prior configuration | Adaptive fusion interface |
|---:|---|---|
| 2 | Global phrase bank; RAG weight `.7`; global retrieval sigmoid midpoint `0.0` | `1.5 × (1 − normalized LM entropy)` |
| 3 | Subject-only phrase bank; RAG weight `1.0`; subject confidence midpoint `.20`; `subject_only=true` | `1.5 × (1 − normalized LM entropy)` |
| 4 | Global plus subject banks; RAG weight `.7`; global midpoint `0.0`; subject confidence midpoint `.20` | `1.5 × (1 − normalized LM entropy)` |

Here normalized LM entropy is `entropy(LM prior) / log(number of grid classes)`. The multiplier is driven by confidence in the LM prior, not classifier or EEG confidence. Paper wording should say **LM-confidence-adaptive fusion** (or specify LM-prior-entropy weighting); “adaptive fusion with calibrated classifier confidence” does not describe this ladder implementation. Rungs 2–4 change multiple factors together relative to the fixed-LM Rung 1. Their contrasts cannot isolate RAG, adaptive weighting, or personalization; see the comparisons below.

## Phrase-bank construction, provenance, and tracking

### Clean Study D banks

The RAG section of `splits/ablation_lock.json` records a deterministic run split: `sha256('phrase-bank-v1:' + run_id) modulo 100`, with remainders `0..19` for the validation portion and `20..99` for the training portion. The clean global bank is built from the clean training pool and excludes the deterministic OOF weight-fit runs and every frozen manifest run. The recorded OOF validation set has 14 runs; the global bank contains 66 unique training phrases.

The locked global-bank SHA-256 is `0826a089022eefa3824e310561bba02f41d76e7bea49dadf2fe1b87a0cbacfb9`; this matches the checked-in `data/rag/phrase_bank_global.csv`. The legacy alias `data/rag/phrase_bank.csv` has the same bytes and hash. The 14 validation run IDs and per-subject IDs are recorded in the tracked `data/rag/phrase_bank_internal_validation_sessions.json` and `data/rag/phrase_bank_internal_validation_sessions_by_subject.json`.

**Build-provenance discrepancy:** the lock says the bank was built from the clean training pool, excluding the 14 OOF weight-fit runs and every manifest run. The tracked no-CLI builder, `scripts/build_phrase_bank_from_registry.py`, instead reads the processed target registry and applies its generic 20% salted run-ID split; its code does not explicitly filter against the clean-pool membership, the OOF fit-run list, or the frozen manifest. The locked CSV hash is verified, but the precise code/data state that generated it is not reconstructable from this builder alone. Treat the lock’s provenance as the declared protocol and the bank file as hash-verified; the build procedure remains **UNRESOLVED**. Do not rerun the generic builder to recreate the locked bank.

The tracked Study D per-subject banks contain 17 files and 69 entries summed across files. Each entry is one whitespace-delimited word: individual banks contain 2–6 unique entries. This current inventory disagrees with the comment in `src/models/rag_predictor.py` and older notes that describe roughly 12–18 words per subject. The difference is **UNRESOLVED**. A possible explanation is that older banks were built before the clean-pool exclusions, but that is **UNVERIFIED** and is not asserted as the cause.

### Study Q banks

The locked data audit says Study Q has five repeated task phrases per subject, encoded as typed grid-key labels rather than conventional natural-language words. The tracked builder `scripts/build_study_q_phrase_banks.py` validates that its Train cache run IDs exactly match its clean Train registry and rejects Test/SE003 runs. The lock says the Q Test labels were not read. The generated Q bank files and build report are ignored/local-only in this checkout, not in the origin tree, so their file hashes and current physical inventory are not source-pinned here. This bank cannot establish natural-language personalization.

### Development and synthetic banks

The public-domain development corpora and their split manifest are tracked. They are four author-labelled sources prepared for software debugging, not participant data. Generated `development_banks` outputs are ignored/local-only. The tracked synthetic validation banks are separately tagged `synthetic_validation`, built from public-domain filler words, and kept outside the Study D subject-bank directory. The synthetic bank builder excludes words in the selected subject’s legacy held-out target list; this is a software-mechanism diagnostic, not a clean-M0 benchmark or participant test.

The generated Study Q banks, generated development banks, and local participant-preparation/dry-run outputs are ignored and absent from origin. The `participant_local` outputs present in the current worktree are the assistant-generated synthetic dry-run sample and its generated banks/split metadata, not participant text. Do not commit text samples or participant banks. The prose dead-arm README mentioned in local notes is also ignored and absent from origin; it is not evidence cited by this chapter.

## Study D ladder results and interpretation

The final RAG ladder artifact is `results/tables/ablation_ladder_clean_m0.json`. Its population is the one-shot frozen Study D manifest: 57 runs, 17 subjects, and 339 covered characters per rung, with zero zero-coverage characters. The table reports artifact summaries, not participant outcomes.

The ladder uses 17 row/column flashes per sequence, with each sequence assigned 2.0 seconds (`2/17` seconds per flash) in `scripts/run_clean_m0_ablation_ladder.py`. Flashes and sequences per character are arithmetic means over covered characters; mean seconds per character is the arithmetic mean of observed flash time. `src/evaluation/sequence_scoring.py::variable_time_itr` computes Wolpaw ITR: for 72 grid classes, accuracy `p`, and observed mean character time `T`, bits per character are `log2(72) + p log2(p) + (1-p) log2((1-p)/71)`, omitting zero-probability terms and flooring at zero; multiply by `60/T` for bits/min. `src/models/decoder.py::calculate_itr` accepts accuracy as a percentage and converts it to a fraction. The ladder reports WPM as ITR/5; `src/evaluation/metrics.py::SpellerMetrics.wpm` instead computes `(60 / mean seconds per character) / 5`, so these WPM definitions are not interchangeable.

| Rung | Study D manifest condition | Accuracy | Flashes/character | Sequences/character | Mean seconds/character | Variable-time ITR (bits/min) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Fixed EEG + LM prior | 75.81% | 62.05 | 3.79 | 7.30 | 31.92 |
| 2 | Adaptive fusion + global RAG | 76.11% | 58.80 | 3.59 | 6.92 | 33.89 |
| 3 | Adaptive fusion + subject-only RAG | 75.52% | 63.04 | 3.86 | 7.42 | 31.24 |
| 4 | Adaptive fusion + gated global/subject RAG | 76.11% | 58.80 | 3.59 | 6.92 | 33.89 |

In the current clean-M0 replay, Rung 4 has 33 argmax flips from Rung 1: 13 moved toward the target and corrected a previously wrong decision; 12 moved away and regressed a previously correct decision. The net is one additional correct character relative to Rung 1, a descriptive wash. Rung 4’s final outputs match Rung 2 exactly in the artifact.

The Rung 4 deferral statistic is 312/339 covered characters (92.04%). In the artifact it is stored at `summaries['4'].overall.flip_diagnostics_vs_rung1.sufficiency_gate_deferred_to_global_share`. The evaluator’s per-character flag is true when `subject_gate_weight <= 1e-12`; it indicates the subject contribution was suppressed and the diagnostic labels that outcome as deferral to the global share. The flag does not separately require a positive global-bank weight, so it is not an isolated measure of global-bank activity or of the sufficiency gate’s causal effect. It reflects the combined subject-match, confidence, phrase-count, and sufficiency gates.

| Paired comparison | Mean paired accuracy difference | Better / tied / worse subjects | Wilcoxon p | 95% subject-bootstrap CI | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Rung 2 vs Rung 1 | −0.29 pp | 5 / 7 / 5 | .646 | −2.71 to +2.06 pp | 1.000 |
| Rung 3 vs Rung 1 | −1.04 pp | 3 / 10 / 4 | .498 | −3.72 to +1.13 pp | 1.000 |
| Rung 4 vs Rung 1 | −0.29 pp | 5 / 7 / 5 | .646 | −2.72 to +2.17 pp | 1.000 |
| Rung 4 vs Rung 2 | 0.00 pp | 0 / 17 / 0 | 1.000 | 0.00 to 0.00 pp | 1.000 |

None of the paired comparisons survives Holm correction. Rungs 2–4 are unsupported bundles on this Study D population; their small descriptive differences do not establish an RAG or personalization benefit. The Rung 2 versus Rung 1 comparison also changes RAG-bank presence, LM-entropy-adaptive weighting form, and maximum weight scale (fixed alpha `.85` versus adaptive scale up to `1.5`), so it cannot attribute an effect to any one change.

**Paper summary:** the fixed LM prior gave a descriptive `+3.79 pp` change, which was not significant after Holm correction (`p=.861`). This is the `1v0` comparison in `results/tables/ablation_ladder_clean_m0.json` (`holm_p=0.861493`). Nothing in Rungs 2–4 significantly improved on the fixed-prior result. Report Rungs 2–4 as RAG bundles and describe the ladder as showing no significant stepwise benefit; do not attribute the fixed-prior difference to RAG or call it calibrated-classifier-confidence adaptation.

## Why Study D and Study Q do not test personalization

| Dataset | Available text | Why it is not prospective personalization evidence | Source |
|---|---|---|---|
| Study D | Fixed prompted task vocabulary; the current tracked per-subject banks contain 2–6 unique one-token entries each | It is not participant-authored usage history. The separately documented “12–18 words per subject” figure is unresolved against the actual locked bank files. | `splits/rag_lock.json`; tracked `data/rag/by_subject/phrase_bank_D_*.csv`; `src/models/rag_predictor.py` |
| Study Q | Five repeated task phrases per subject, encoded as typed grid-key labels | The sequences are neither conventional natural-language text nor enough varied user-authored history to test personalization. | `splits/rag_lock.json` (`available_data_audit` and `current_gate`) |

No participant text exists in the committed evidence. The lock records participant access as pending and `natural_user_corpus_available=false`. No personal corpus result may be inferred from Study D, Study Q, synthetic text, or public-domain authors.

## Public-domain development benchmark and software diagnostics

`results/tables/rag_development_benchmark.json` is a development-only benchmark. The four author-labelled public-domain texts are split chronologically: first 60% for the bank, next 20% for training-only gate tuning, and final 20% for evaluation. The benchmark caps each author’s source at 4,000 whitespace tokens, scores 24 held-out character cases per author (96 total), and uses 8 gate-tuning cases per author. The evaluation text is not added to any bank.

| Development population: macro across four authors, 24 held-out character cases each | LM-only NLL (nats/character) | Global-bank NLL | Author-bank NLL | Gated-blend NLL | LM-only top-3 | Global top-3 | Author top-3 | Gated top-3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Public-domain author-separated smoke benchmark | 4.463 | 4.151 | 4.245 | 3.763 | .323 | .448 | .458 | .479 |

Per-author NLL (each value uses 24 held-out character cases for that author):

| Author | LM-only | LM + global bank | LM + author bank | Gated blend |
|---|---:|---:|---:|---:|
| Arthur Conan Doyle | 3.800 | 2.917 | 3.248 | 2.902 |
| Herman Melville | 4.739 | 5.052 | 5.227 | 4.723 |
| Jane Austen | 4.901 | 4.113 | 4.508 | 3.587 |
| Mary Shelley | 4.411 | 4.522 | 3.995 | 3.839 |

Global-bank NLL degrades for Melville and Mary Shelley relative to LM-only. Author-bank NLL degrades for Melville. The lock’s worst-case bound permits at most `.1` NLL increase in nats per character and `.05` top-1 accuracy drop. In this development population, the author-bank Melville NLL increase is `.489` nats/character and the Mary Shelley top-1 accuracy drop is `.125`; the bounded-degradation gate fails. In the separate development-only topic-switch diagnostic, an Austen training bank followed by Melville text has LM-only NLL `4.739` and subject-bank NLL `5.217` (24 held-out Melville cases).

The artifact’s exact `claim_warning` is: “All outputs debug the offline code path. They are not evidence of participant personalization benefit.” Its status is `development_only_completed`; its source is public-domain text, not participant data. The sample is capped at 96 held-out cases and supports no user benefit or clinical claim.

The lock’s development gate grid was RAG weights `.25/.7`, sufficiency midpoints `32/128` tokens, subject-confidence midpoints `.2/.4`, bigram minimum counts `2/5`, and normalized-entropy ceilings `.4/.8`; the benchmark artifact records its training-only selected gate values by author. Its learning-curve bank sizes were `0, 64, 128, 256, 512, 1024` words. The lock asks for a topic-switch report over the first 100 eligible characters without retuning; the capped diagnostic artifact instead contains 24 held-out Melville cases, so the full 100-character protocol was not exercised.

The same development artifact contains learning curves and cold-start update curves. The synthetic end-to-end dry-run artifact, `results/tables/rag_self_generated_dry_run.json`, records one assistant-generated 2,030-token sample split into 1,218 bank-training, 406 gate-tuning, and 406 evaluation tokens, then scores 12 next-key priors; every prior was finite and normalized, and the evaluation tail was not added to the bank. These are software checks only. OOV, special-key, backspace, empty-bank, and adversarial-bank cases are represented in the RAG tests and diagnostics; this session did not run those tests, so a fresh test-pass claim is **NOT RUN** here.

## Participant text protocol — Phase 2b

The full lock is `splits/rag_lock.json`. It is preregistered, not executed. Its participant collection and analysis requirements are:

| Requirement | Locked value |
|---|---|
| Source | Consented participant-authored text, or participant-confirmed decoded text collected before evaluation; one naturally authored sample in the speller language, with date range and topic documented |
| Minimum sample | 700 whitespace tokens total |
| Chronological split | First 60% for bank and gate training, next 20% for gate tuning, final 20% held out; minimums 420/140/140 tokens |
| Ordering | Sort by timestamp when available; otherwise preserve participant-supplied document order; timestamped JSONL records contain timestamp and text |
| Anonymization | Before import; automatically redact email and phone patterns; participant reviews and redacts names and selected sensitive spans |
| Storage | Pseudonymous local-only files; never commit source text or phrase banks; no cloud upload required; no text logging |
| Bank use | Build from the first 60%, fit gates using only the following 20%, freeze before session, never seed from the held-out 20% |
| Compared models | Base LM; LM + global bank; LM + subject bank; gated blend |
| Primary metrics | Mean next-character NLL in nats on eligible grid characters; top-k accuracy at k=1/3/5; true-character mean rank and mean reciprocal rank; keystroke savings (fraction of held-out characters with the true key in top 3) |
| Secondary metrics | Learning curve at 0/64/128/256/512/1024 training tokens; cold-start/cumulative personalization; first 100 eligible characters after a topic switch, without retuning; OOV/special-key, backspace, empty-bank, and adversarial-bank robustness |
| Inference | Participant is the unit; aggregate token-level metrics within participant; paired participant-level comparisons, Holm correction, and participant bootstrap 95% percentile intervals |
| Primary family | LM+global vs LM-only; LM+subject vs LM-only; gated blend vs LM-only; Holm across the three comparisons |
| Secondary correction | Holm across three RAG comparisons and three secondary metric families |
| Degradation bound | Maximum NLL increase `.1` nats per character and maximum top-1 accuracy drop `.05` |
| Online adaptation | Teacher-forced decoded-prefix updates only; score each next key before update; report cold-start and cumulative updates |

`scripts/prepare_participant_text_sample.py` implements UTF-8 text/JSONL ingestion, timestamp ordering when **all** JSONL records have timestamps, automatic email/phone replacement, the minimum-token check, chronological 60/20/20 splitting, and a phrase bank built from the first split only. The parser writes a `splits.json` containing the three split texts—including the held-out evaluation text—and a bank CSV in the caller-selected output directory. Therefore that directory must be local-only and protected from commits. Names and participant-selected sensitive spans still require review before import; the script does not redact names automatically. The lock and closeout report say participant access is pending, no participant text has been accessed, and offline testing is not live evidence. No participant metrics, paired comparisons, or intervals have run.

The lock predates natural-text evaluation and participant-text access. Its development claim is explicitly disallowed, and synthetic results are explicitly not primary evidence. The locked G2 decision is **no-go for live RAG or a natural-text personalization claim**. The simulator’s D known-result check failed with absolute accuracy error `.179` against tolerance `.05`; Q passed its check. The simulator record is `results/tables/eeg_simulator_loso_validation.json`. Phase 4 therefore retains EEG-only and fixed LM fusion as the evaluated arms, excludes RAG from live sessions pending participant-text G2, and keeps live flashing uniform.

## Experiment and evidence register

| ID | Question | Method and population | Result | Artifact / commit | Branch | Status |
|---|---|---|---|---|---|---|
| R6.1 | Does global LM+RAG help versus LM-only in an offline smoke? | Four public-domain authors; chronological split; 24 held-out characters per author | Macro NLL/top-3 shown above; per-author degradation and bound failure; no participant inference | `results/tables/rag_development_benchmark.json`; `62ce804` | `optimise` | DEV-ONLY |
| R6.2 | Does RAG produce valid normalized priors in a synthetic end-to-end dry run? | Assistant-generated synthetic repeated prose; 12 next-key priors | All priors finite/normalized; evaluation tail not added to bank | `results/tables/rag_self_generated_dry_run.json`; `62ce804` | `optimise` | DEV-ONLY |
| R6.3 | Does global RAG improve the clean-M0 Study D manifest ladder? | One-shot manifest; 57 runs / 17 subjects / 339 covered characters per rung | Rung 2 76.11% vs Rung 1 75.81%; paired 2v1 −0.29 pp, Holm p=1.000; bundle confound | `results/tables/ablation_ladder_clean_m0.json`; `splits/ablation_lock.json`; `e7b133c` | `optimise` | FINAL replay; no significant benefit |
| R6.4 | Does subject-only RAG improve that manifest population? | Same one-shot Study D manifest population | Rung 3 75.52%; paired 3v1 −1.04 pp, Holm p=1.000 | Same ladder artifact and lock | `optimise` | FINAL replay; no significant benefit |
| R6.5 | Does gated global/subject RAG improve that manifest population? | Same one-shot Study D manifest population | Rung 4 76.11%; 33 flips (13 toward, 12 away), net +1 correct character vs Rung 1; paired 4v1 Holm p=1.000; exact output match to Rung 2 | Same ladder artifact and lock | `optimise` | FINAL replay; no significant benefit |
| R6.6 | Is subject text sufficient for prospective personalization? | Participant text, minimum 700 tokens and locked chronological split | No participant text exists; participant access pending | `splits/rag_lock.json`; `docs/phase1-3_closeout.md`; `62ce804` | `optimise` | PENDING / G2 NO-GO |
| R6.7 | How does the gate behave across synthetic corpus sizes with real Study D EEG? | Legacy held-out D subject/pooled mechanism diagnostics; synthetic banks only | Gate-mechanism smoke outputs exist, but are not clean-M0 personalization evidence | `results/tables/sufficiency_gate_sweep.json`; `results/tables/sufficiency_gate_pooled_sweep.json`; `dbe55ed` | `optimise` | SUPERSEDED mechanism diagnostics |
| R6.8 | What does the older Study D RAG sweep report? | Old SWLDA pipeline; script docstring specifies held-out test split | Legacy values are not valid current held-out evidence; README’s train-split-selection statement conflicts with the script | `scripts/sweep_rag_params.py`; `results/tables/rag_sweep.json`; `README.md`; `ac0fab1` | `optimise` | DEAD / SUPERSEDED |
| R6.9 | What did the older personalization replay report? | Historical contaminated SWLDA base and pre-correction held-out tuning | Audit-only values; see dead-arms table; not current evidence | `results/tables/personalization_ablation_studyd.json`; `results/tables/personalization_ablation_studyd_status.json`; `2952a71` | `optimise` | SUPERSEDED / AUDIT ONLY |
| R6.10 | Can the participant preparation utility process the preregistered text protocol? | Participant text would be read; none is available or accessed | Parser and implementation inspected; no participant run | `scripts/prepare_participant_text_sample.py`; `splits/rag_lock.json` | `optimise` | PENDING / NOT RUN |
| R6.11 | Does adding the fixed LM prior help in inner OOF fusion? | Study D clean training pool; nested run-grouped OOF, 83 runs / 17 subjects / 498 characters | Classifier inner-fusion artifact; OOF only, not manifest evidence | `results/tables/classifier_inner_fusion.json`; `results/tables/classifier_inner_fusion_stats.json` | `optimise` | COMPLETED / OOF ONLY |
| R6.12 | Which fixed alpha minimizes sequence NLL for the LM-only Rung 1 prior? | Deterministic 20% salted-SHA-256 split; 14 validation runs / 84 characters | 101-point alpha grid, 0–5 by .05; selected alpha `.85`, OOF sequence NLL `.296513` | `results/tables/ablation_oof_weights.json`; `splits/ablation_lock.json` | `optimise` | COMPLETED / FIT ONLY |
| R6.13 | Which stopping policy was selected at alpha `.85`? | Clean-M0 nested run-grouped OOF; 498 characters | `tau=.80`, min 2, max 10; 84.34% stopped accuracy, 4.91 mean sequences | `results/tables/ablation_inner_m0_policy.json`; `splits/ablation_lock.json` | `optimise` | LOCKED / OOF SELECTION |
| R6.14 | What policy candidate results at alpha `.1`? | Clean-M0 nested run-grouped OOF; selection/scoring on same 498-character population | Candidate `tau=.70`, min 2, max 10; 81.93% accuracy under either candidate or locked `.80`; not selected for/evaluated on Study D manifest | `results/tables/ablation_inner_m0_policy_alpha_0p1.json`; `splits/finalist_lock.json` | `optimise` | CANDIDATE ONLY |

## Dead arms and superseded results

| Dead arm / historical note | Record | Status and limitation |
|---|---|---|
| `sweep_rag_params.py` fixed RAG sweep | `results/tables/rag_sweep.json`; `README.md` historical section | **DEAD / SUPERSEDED.** The script docstring says held-out test split and the implementation loops over the old SWLDA pipeline. The README’s claim that the grid was selected on a train split is stale and conflicts with the script. The historical held-out SWLDA README table lists fixed fusion at 54.15% and baseline at 53.30%, but its cited aggregate CSV is absent from origin; do not treat the pair as independently artifact-verified or current evidence. |
| Old subject-aware RAG replay | `results/tables/personalization_ablation_studyd.json`; status at `results/tables/personalization_ablation_studyd_status.json` | **SUPERSEDED_AUDIT_ONLY.** It used the contaminated historical SWLDA base and pre-correction held-out tuning. The artifact has historical Rung 4 flip counts of 31 total, 12 toward target, and 2 away. Its Rung 1→Rung 4 mean paired difference was +0.490 pp (Wilcoxon p=.317, Holm p=1.0); Rung 1→Rung 3 was +0.245 pp (p=.317, Holm p=1.0). Keep these figures here only as historical audit records. |
| Historical RAG ladder/personalization uplift claims | Status artifact `personalization_ablation_studyd_status.json` | **SUPERSEDED.** Do not use old Rung 3/4 effects as current evidence. Current clean-M0 Rung 4 diagnostics are in the manifest-ladder section above. |
| Legacy sufficiency sweeps | `results/tables/sufficiency_gate_sweep.json`; `results/tables/sufficiency_gate_pooled_sweep.json`; history before the clean-M0 lock | **SUPERSEDED mechanism diagnostics.** They use synthetic banks with real held-out Study D EEG and predate the current clean-M0 ladder. They demonstrate implementation behavior only, not participant benefit or a clean held-out estimate. |
| Held-out-tuned stopping sweep | `results/tables/dynamic_stopping_threshold_sweep.json`; historical split/leakage records | **DEAD / SUPERSEDED.** Policy selection used held-out Study D sessions; replaced by clean training-only nested OOF selection. |
| First toy-bank / silent-fallback incident | Unrecorded project note; no matching committed artifact or history entry found | **UNVERIFIED.** The recollection says the bank had about 12 phrases and near-zero vocabulary overlap. No causal account is established here. |
| First-token prior arm | Historical implementation/experiment note; no current locked ladder arm | **DEAD / SUPERSEDED.** The current RAG ladder predicts from matched phrase continuations and does not use a first-token-prior arm; no claim about its standalone benefit is made here. |
| Old Rung 3/4 regression and “bwards” / “friene” fragmented-bank examples | Unrecorded project note; no matching strings or controlled comparison found in committed code/artifacts | **UNVERIFIED.** An older roughly −6.4 pp drop and its attribution to ungated bigram backoff are recollections, not established results. The current code includes count and entropy gates, but that does not verify the alleged cause or examples. |

## Final locked configuration and live-session scope

| Lock | SHA-256 | Relevant frozen fields | Commit |
|---|---|---|---|
| `splits/rag_lock.json` | `169b758421233410b2ec0ff6c5cf348a29cbf22f276c624a94e5f42394937fdd` | Participant protocol, clean bank split, 14 weight-fit validation runs, 66 global training phrases and hash, sufficiency/backoff parameters, development-only benchmark protocol, and G2 gate decision | `62ce804` (2026-10-08) |
| `splits/ablation_lock.json` | `840c12db474735757ece08ae70b489795870c09933dfd18819a34a32de895095` | Clean M0; Rung 1 fixed alpha `.85`; Rungs 2–4 LM-entropy adaptive scale `1.5`; RAG weights `.7/1.0/.7`; global bank hash; stopping `tau=.80`, min 2, max 10 (stop after at least 2 sequences when max posterior ≥ tau, otherwise at 10) | `8cb0649` (2026-10-07) |

The checked-in global bank hash matches the value in the RAG and ablation locks. The D ladder’s live-session scope remains RAG-disabled: Phase 4 allows the EEG-only and fixed LM-fusion research arms, keeps uniform flashing, and excludes RAG until participant-text G2 passes.

The dated listing for **`git ls-remote --tags origin`, 2026-10-26** records the following status. This is a dated listing, not a live query in this session.

| Tag | Listing status | Local tag target |
|---|---|---|
| `ablation-final` | Origin-listed | `494066a` |
| `baseline-d-final` | Origin-listed | `88d19ec` |
| `classifier-final` | Origin-listed | `dbf5293` |
| `preproc-ablation-final` | Origin-listed | `b9e9266` |
| `phase3-final` | Local-only | `62ce804` |
| `q-inner-final` | Local-only | `f5010b0` |
| `q-manifest-frozen` | Local-only | `449c266` |

The RAG implementation history begins at `8525345` (initial predictor), proceeds through subject-aware retrieval (`c2c7b98`), confidence-gated backoff (`eb18a90`), leakage-safe bank diagnostics (`dbe55ed`), and gated diagnostics (`2952a71`), then reaches the Phase 1–3 no-go closeout at `62ce804`. Branch inspection found no RAG feature required from the older branches that is missing on `optimise`: `CNN` has an earlier predictor; `codex/create-detailed-project-documentation-twto65` starts from the initial RAG insertion; `codex/personalized-rag-focus-ui` has an earlier implementation without current token-mode/update support and uses the historical evaluator; `SKB-J8` has no RAG implementation files. These older branch states are not evidence for current results.

## Limitations and open items

* No participant text exists; participant access is pending. There is no participant-level RAG estimate or personalization claim.
* The development benchmark is a 96-case debugging diagnostic. It has per-author degradation and fails the locked bounded-degradation check; its warning explicitly disallows participant-benefit inference.
* Study D banks are fixed task vocabulary, and their current per-subject inventory conflicts with the older 12–18 word statement. The discrepancy is unresolved.
* Study Q banks are five repeated typed-key phrases per subject, not conventional language history.
* Study D Rungs 2–4 are multi-factor bundles; no significant benefit is supported, and their effects cannot be separated.
* The specific toy-bank incident and fragmented-word examples remain unverified.
* The old README parameter-sweep statement conflicts with the actual held-out sweep script. The README’s old result table cites a CSV absent from origin.
* Generated Q banks, development banks, and participant-local outputs are ignored/local-only; no hash is available for them in origin.
* Current unit-test files are not evidence of a passing test run in this session. Test execution is **NOT RUN**.

## Reproduction and script interfaces

The following scripts have **no argparse CLI**. Their inputs/outputs below come from the checked-in source. No commands are provided for them.

| Script | Inputs | Outputs / effect | Status |
|---|---|---|---|
| `scripts/build_phrase_bank_from_registry.py` | Processed Study D target registry; deterministic run split | Global and legacy-alias phrase banks, per-subject banks, validation-session manifests, subject-bank size report | No CLI; historical builder; do not rerun against sealed/current labels as a routine step |
| `scripts/build_study_q_phrase_banks.py` | Clean Q Train registry, classifier cache, grid layout | Generated Q global/per-subject banks and build report in ignored/local output | No CLI; uses training-only sources; generated banks are not in origin |
| `scripts/prepare_rag_development_corpora.py` | Tracked public-domain source texts and split manifest settings | Prepared author-separated development corpora and manifest | No CLI; development-only |
| `scripts/run_rag_development_benchmark.py` | Prepared development corpora, phrase banks, locked benchmark settings | `results/tables/rag_development_benchmark.json` | No CLI; existing artifact is development-only |
| `scripts/run_rag_self_generated_dry_run.py` | Built-in/generated synthetic repeated prose | Synthetic local bank/split outputs and `results/tables/rag_self_generated_dry_run.json` | No CLI; software dry run only |
| `scripts/sweep_rag_params.py` | Old SWLDA model, processed Study D data, old held-out test split, legacy global bank | `results/tables/rag_sweep.json` | No CLI; **DEAD / SUPERSEDED**, held-out sweep |

Verified parsers:

| Script | Actual argparse options | Data-access boundary |
|---|---|---|
| `scripts/prepare_participant_text_sample.py` | Positional `source`, positional `output_dir`, `--min-tokens` (default `700`) | Reads participant text if invoked; none exists, so **NOT RUN**. Caller must choose an ignored local-only output directory. |
| `scripts/fit_ablation_weights.py` | `--cache` (default `data/cache/classifier_1_12_no_baseline.h5`), `--scores` (default `results/tables/classifier_nested_calibration_scores.npz`), `--output` (default `results/tables/ablation_oof_weights.json`) | Parser verified. Existing fit artifact is recorded above; fitting is not rerun here. |
| `scripts/reselect_m0_policy_alpha_0p1.py` | `--output` (default `results/tables/ablation_inner_m0_policy_alpha_0p1.json`) | Parser verified; it reads the fixed score/cache paths in source. Existing candidate artifact is recorded above; selection is not rerun here. |
| `scripts/build_synthetic_validation_bank.py` | `--subject` (required), `--sizes` (one or more integers, required), `--seed` (default `42`) | Reads held-out target words to exclude them and writes synthetic banks; not for routine rerun. |
| `scripts/sweep_sufficiency_gate.py` | `--subject` (required), `--sizes` (one or more integers, required) | Uses legacy Study D held-out EEG; artifact is superseded, do not rerun for model selection. |
| `scripts/sweep_sufficiency_gate_pooled.py` | `--subjects` (one or more; default subject list), `--sizes` (one or more; default size list) | Uses legacy Study D held-out EEG; artifact is superseded, do not rerun for model selection. |
| `scripts/run_personalization_ablations.py` | `--debug-subject`, repeated `--subject`, `--reset-results`, `--force`, `--disable-bigram-backoff`, `--bigram-min-count`, `--bigram-max-normalized-entropy`, `--allow-heldout-eval`, `--tau`, `--min-sequences`, `--max-sequences`, `--debug-trace-calls`, `--output` | Requires held-out authorization and uses the historical classifier path; **SUPERSEDED**, not for routine rerun. |

No one-shot held-out evaluator is a reproduction command. The participant preparation parser is documented for future use only; this chapter does not invoke it.

## Component 7 handoff — scheduler / system chapter

```text
1. `RAGPredictor.predict_next_char(context)` returns a normalized grid prior; RAG wraps the base LM prior before EEG fusion.
2. Clean Study D Rungs 2–4 use global, subject-only, and global/subject banks with RAG weights .7, 1.0, and .7 respectively.
3. The subject-data sigmoid lock is midpoint 15 tokens, sharpness .20; the subject-confidence midpoint is .20.
4. The bigram backoff lock is minimum successor count 2 and maximum normalized entropy .65; token mode disables character-bigram backoff.
5. Global bank SHA-256 is `0826a089022eefa3824e310561bba02f41d76e7bea49dadf2fe1b87a0cbacfb9`.
6. Rung 4 suppresses the subject contribution for 312/339 Study D manifest characters (92.04%); this is a combined-gate diagnostic.
7. Rungs 2–4 are unsupported Study D bundles; no paired comparison survives Holm correction.
8. No participant-authored text exists; Phase 2b is pending and G2 is no-go for live RAG.
9. Current Rung 4 flips are 33 total, 13 toward target and 12 away; net accuracy change versus Rung 1 is one character.
10. Retain uniform flashing and the EEG-only/fixed-LM research arms; do not make RAG, personalization, or scheduler-benefit claims.
```
