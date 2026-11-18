# Component 5 — Language-model prior, fusion, stopping, and clean-M0 ablations

This chapter documents the language-model (LM) prior, Bayesian-style fusion, sequential decoder, and the locked clean-M0 ablation ladder. It is based on the committed implementation and artifacts on branch `optimise`. Components 2–4 (preprocessing, split/leakage protocol, and classifier) are treated as fixed inputs. Retrieval-augmented-generation (RAG) internals and personalization gates are Component 6; only their ladder interfaces and result-level behavior are summarized here.

## Scope and evidence populations

The results below come from different populations and must stay separate:

| Evidence | Population and evaluator | Use |
|---|---|---|
| Fusion-weight fit | Clean training-pool OOF, deterministic 20% salted-SHA-256 run validation split; 14 runs and 84 validation characters; artifact `results/tables/ablation_oof_weights.json` | Fit ladder fusion weights only |
| Stopping-policy selection | Clean-M0 nested run-grouped OOF; 498 characters; artifact `results/tables/ablation_inner_m0_policy.json` | Select ladder stopping policy |
| Study D ladder | One-shot frozen Study D manifest; 57 runs, 17 subjects, 339 covered characters per rung, no zero-coverage characters; artifact `results/tables/ablation_ladder_clean_m0.json` | Final rung comparison |
| Study D classifier manifest | Separate one-shot classifier evaluator; 338 total characters, including 11 zero-coverage terminal characters; artifact `results/tables/classifier_manifest_eval.json` | Classifier evaluation, not the ladder denominator |
| Study Q inner comparison | Nested subject-session OOF; 2,592 characters; artifact `results/tables/study_q_nested_fusion.json` | Study Q training-only fusion evidence |
| Study Q test-once | Frozen SE003 Test manifest, 180 runs, 36 subjects, 1,296 as-recorded characters; artifact `results/tables/study_q_test_once.json` | Separate external, one-shot result |

The 339-versus-338 Study D character-count difference is not reconciled by these artifacts: **UNVERIFIED**. They use different evaluators and denominator policies; do not combine or treat them as a single denominator. The ladder reports zero zero-coverage characters, while the classifier manifest evaluator includes 11.

## LM prior over grid keys

`src/models/llm_predictor.py` (`LLMPredictor`) converts a context-conditioned causal-LM score into a vector aligned with the spelling grid. Its default model is `distilgpt2`; multi-character grid labels are rendered using `configs/llm/special_key_map.json`. Ordinary keys are lowercased. Empty context returns a uniform grid prior.

For a non-empty context, the implementation lowercases it and changes underscores to spaces, then separates the completed prefix from the current partial word at the last space. For every grid key it forms the full candidate completion from that partial word plus the rendered key. `_continuation_log_probabilities` tokenizes the prefix and complete prefix-plus-candidate string, scores every candidate token conditionally, and sums those token log-probabilities. If tokenization does not preserve the prefix token sequence, it falls back to appending separately tokenized candidate IDs. Candidate scores are exponentiated after subtracting the largest finite score and normalized over the grid. Non-finite candidates receive zero; if none are finite, the implementation returns a uniform prior.

### Resolved prior-method discrepancy

The current committed code uses **full-string continuation scoring**. The descriptions in README sections 1.2 and 3.8 that describe summing next-token probability mass over vocabulary tokens matching a candidate prefix are stale and contradict the current implementation. The current docstring explicitly says full-string continuation scoring replaced the former first-token prefix lookup, and the change is present in commit `fe044b6` (`prepare clean M0 ablation ladder before lock`), together with the special-key map. Earlier code in `3f6fe42` used a one-token-per-key mapping and selected logits for the mapped first token; this is the superseded implementation. The stated code history does not establish a measured performance difference between the methods: **NOT RUN** (no verified before/after artifact found).

The algorithm therefore scores the conditional probability of each *complete mapped key continuation*, not a marginal sum over all vocabulary tokens beginning with the candidate prefix. README sections 1.2 and 3.8 should be updated to match this chapter when those sections are next edited.

## Fusion and placement in decoding

`src/models/fusion.py` defines `BayesianFusionEngine`. For EEG class probabilities \(p_E(c)\), LM probabilities \(p_L(c)\), and weight \(\alpha\), `fuse()` computes and normalizes:

```text
log unnormalized posterior(c) = log clip(p_E(c), epsilon, 1)
                              + alpha * log clip(p_L(c), epsilon, 1)
posterior(c) = exp(log unnormalized posterior(c) - logsumexp(all classes))
```

Before logging, classes where `llm_probs == 0` are marked unmapped and their LM log-probability is set to negative infinity. They are excluded from the normalized posterior rather than epsilon-clipped. EEG probabilities are epsilon-clipped. If the engine is toggled off, `fuse()` returns the EEG probabilities unchanged.

In fixed mode, `alpha = base_alpha`. In adaptive mode, the code sets `alpha = base_alpha * entropy(eeg_probs + epsilon) / log(num_classes)`. Thus the adaptive multiplier is driven by EEG uncertainty: it increases with normalized EEG entropy. The normalization assumes `num_classes` equals the actual grid size.

The current sequential-decoder path uses `get_initial_log_bias()`: it computes `alpha * log(LM prior)` once as a starting log-posterior bias, then adds each valid sequence's log EEG grid posterior. The ladder replay applies this initial bias before per-sequence accumulation. The Study Q nested-fusion artifact explicitly records `applied_once_before_sequence_accumulation: true`. The alpha-.1 Study D classifier evaluation likewise applies the prior as an initial sequence bias. Do not describe the prior as independently re-fused after each sequence.

One implementation distinction matters: `get_initial_log_bias()`'s adaptive mode computes its scale from **LM entropy**, using `base_alpha * (1 - normalized LM entropy)`, whereas `fuse()`'s adaptive mode computes its scale from **EEG entropy**. The clean-M0 ladder calls `get_initial_log_bias()` (see `scripts/run_clean_m0_ablation_ladder.py`); its adaptive rung results must be interpreted using the LM-entropy form. These are two distinct code paths, not one interchangeable adaptive formula.

## Sequential decoder and stopping policy

`src/evaluation/sequence_scoring.py` is the shared sequence-level implementation. Row/column flash scores are aggregated into a grid posterior. The sequential decoder initializes the log posterior from the prior bias, adds `log(clip(sequence_grid_posterior, 1e-9, 1))` per valid sequence, and normalizes with softmax for confidence. It stops when at least the minimum number of sequences has been consumed **and** maximum posterior confidence is at least `tau`; otherwise it stops when the maximum-sequence limit is reached. The lock contains a `legacy_rc_clamp=15` field, but repository code does not reference that field; the ladder evaluator's active maximum-sequence policy is 10. Treat the legacy field as locked metadata, not an applied clamp.

The active ladder policy is `tau=0.80`, minimum 2 sequences, maximum 10. It was selected on clean-M0 nested run-grouped OOF only: 498 characters, fused accuracy 84.34% at alpha .85, mean 4.91 sequences, and zero decision flips relative to the default policy. These are OOF policy-selection results, not manifest performance.

Two alpha values appear in different stages and answer different questions:

* **Alpha .85** is the fixed LM weight fitted for ladder Rung 1 on the 14-run/84-character deterministic OOF weight-fit split. The tau=.80 stopping policy above was selected at alpha .85 and used in the locked ladder replay.
* **Alpha .1** is the fixed fusion weight in `splits/finalist_lock.json` and the classifier manifest evaluation. An alpha-.1 training-OOF re-selection candidate chose tau=.70 (min 2, max 10), with 81.93% accuracy under either the candidate or locked tau=.80 policy; mean sequences fell from 6.45 to 5.73 and 14 decisions changed (6 corrected, 6 regressed, 2 wrong-to-wrong). This candidate is recorded in `results/tables/ablation_inner_m0_policy_alpha_0p1.json`; it was **not** selected for or evaluated on the Study D manifest. The manifest therefore reports its locked alpha .1 using the policy selected at alpha .85. Do not present the alpha-.1 candidate as a manifest-tested policy.

## Clean-M0 ablation ladder

`splits/ablation_lock.json` freezes clean M0, the Rung 1 fixed alpha .85, adaptive weights, RAG interfaces, and stopping policy. The weight-fit artifact records 14 validation runs and 84 characters, a small fit set that limits confidence in the chosen weights. Rungs 0 and 1 isolate the EEG-only to fixed-LM-prior step. Rungs 2–4 are reported at their locked interface and outcome level; they invoke `RAGPredictor`, whose internals are documented separately in Component 6.

| Rung | Locked condition | Manifest accuracy | Flashes/character | Sequences/character | Mean seconds/character | Variable-time ITR (bits/min) |
|---:|---|---:|---:|---:|---:|---:|
| 0 | EEG only; alpha 0 | 70.80% | 80.22 | 4.92 | 9.44 | 22.27 |
| 1 | Fixed EEG + DistilGPT-2 prior; alpha .85 | 75.81% | 62.05 | 3.79 | 7.30 | 31.92 |
| 2 | Adaptive fusion + calibrated global RAG; adaptive weight 1.5, RAG weight .7 | 76.11% | 58.80 | 3.59 | 6.92 | 33.89 |
| 3 | Adaptive fusion + subject-only RAG; adaptive weight 1.5, RAG weight 1.0 | 75.52% | 63.04 | 3.86 | 7.42 | 31.24 |
| 4 | Adaptive fusion + gated global/subject RAG blend; adaptive weight 1.5, RAG weight .7 | 76.11% | 58.80 | 3.59 | 6.92 | 33.89 |

For adaptive Rungs 2–4, the effective prior weight is `1.5 × (1 − normalized LM entropy)`, where normalized LM entropy is `entropy(LM prior) / log(number of grid classes)`. The scale therefore follows confidence in the LM prior, not classifier or EEG confidence. In paper prose, describe this as **LM-confidence-adaptive fusion** (or specify that the weighting is driven by LM prior entropy). Do not call this ladder implementation “adaptive fusion with calibrated classifier confidence”: that wording does not match the code. The separate `fuse()` method's adaptive branch uses EEG entropy, but it is not the path used by this ladder.

Each rung's metric population is the same 57-run/17-subject Study D manifest, 339 covered characters, zero zero-coverage characters (`results/tables/ablation_ladder_clean_m0.json`). ITR uses observed variable character time. Rung 4's output matches Rung 2 exactly in the artifact. Its recorded deferral flag is true for 312/339 covered Rung 4 characters (92.04%; `summaries['4'].overall.flip_diagnostics_vs_rung1.sufficiency_gate_deferred_to_global_share`). These are descriptive manifest summaries; paired per-subject inference is the comparison basis.

The predeclared paired family is 1v0, 2v1, 3v1, 4v1, and 4v2, with per-subject Wilcoxon tests, Holm correction, and subject bootstrap confidence intervals. No comparison survives Holm correction:

| Comparison | Mean paired accuracy difference | Better / tied / worse subjects | Wilcoxon p | 95% subject-bootstrap CI | Holm p |
|---|---:|---:|---:|---:|---:|
| 1 vs 0 | +3.79 pp | 8 / 4 / 5 | .172 | −1.29 to +9.08 pp | .861 |
| 2 vs 1 | −0.29 pp | 5 / 7 / 5 | .646 | −2.71 to +2.06 pp | 1.000 |
| 3 vs 1 | −1.04 pp | 3 / 10 / 4 | .498 | −3.72 to +1.13 pp | 1.000 |
| 4 vs 1 | −0.29 pp | 5 / 7 / 5 | .646 | −2.72 to +2.17 pp | 1.000 |
| 4 vs 2 | 0.00 pp | 0 / 17 / 0 | 1.000 | 0.00 to 0.00 pp | 1.000 |

Rung 2 versus Rung 1 changes three things at once: it adds a global RAG bank, changes fixed weighting to LM-entropy-adaptive weighting, and changes the weight scale from fixed alpha .85 to an adaptive scale with a maximum of 1.5. The locked ladder has no RAG-free adaptive rung, and the searched artifacts contain no isolated adaptive-only manifest comparison. Therefore the 2v1 contrast confounds the RAG bank, weighting form, and maximum weight scale; it cannot attribute any difference to one of them. Rungs 3–4 are also RAG configurations, not isolated tests of personalization mechanics. RAG sufficiency/backoff gates and personalization interpretation belong to Component 6.

**Paper-safe summary:** On the one-shot Study D manifest, the fixed LM prior (Rung 1 vs Rung 0) had a mean paired per-subject accuracy difference of **+3.79 percentage points**, which was not statistically significant after the prespecified Holm correction (**Holm p=.861**). The subsequent RAG bundle arms (Rungs 2–4) showed no significant improvement over the fixed-LM rung; no paired comparison in the prespecified family survived correction. Describe Rungs 2–4 as bundles and do not attribute their outcomes to adaptive weighting or personalization separately.

The legacy Study D fused headline (58.9%), old ladder/personalization results, Focus Scan, and the stopping sweep tuned on held-out data are superseded. They must not be used as current held-out estimates; see `docs/03_splits_and_leakage.md` and `dead_arms/README.md`.

### Final locked configuration

| Lock file | SHA-256 | What it freezes | Lock commit |
|---|---|---|---|
| `splits/ablation_lock.json` | `840c12db474735757ece08ae70b489795870c09933dfd18819a34a32de895095` | LM `distilgpt2`, full-string scoring with `configs/llm/special_key_map.json`; fixed Rung 1 alpha .85; adaptive Rungs 2–4 weight scale 1.5 (effective weight from LM entropy); stopping `tau=.80`, min 2, max 10; clean M0 and RAG rung interfaces | `8cb0649` (2026-10-07) |

The final lock SHA-256 was recomputed from the checked-in file. The lock's `created_git_hash` field is `fe044b6`; the current lock revision was committed at `8cb0649`. The ladder artifact records completion at `e7b133c` (2026-10-08).

### Experiment register

| ID | Question | Population / evaluator | Artifact | Status |
|---|---|---|---|---|
| E5.1 | Does a fixed LM prior improve clean-M0 nested OOF sequence outcomes at alpha .1? | Study D clean training pool; nested run-grouped OOF, 83 runs / 17 subjects / 498 characters | `results/tables/classifier_inner_fusion.json` and `results/tables/classifier_inner_fusion_stats.json` | Completed; OOF only, not manifest evidence |
| E5.2 | What fixed alpha minimizes sequence NLL for the LM-only Rung 1 prior? | Deterministic 20% salted-SHA-256 split of clean training-pool OOF; 14 runs / 84 validation characters | `results/tables/ablation_oof_weights.json` | Completed; 101-point alpha grid from 0 to 5 by .05; selected alpha .85, OOF sequence NLL .296513 |
| E5.3 | Which stopping policy is selected at alpha .85? | Clean-M0 nested run-grouped OOF; 498 characters | `results/tables/ablation_inner_m0_policy.json` | Completed; tau .80, min 2, max 10; 84.34% stopped accuracy and 4.91 mean sequences |
| E5.4 | Which policy would be selected for the classifier lock's alpha .1? | Same clean-M0 nested run-grouped OOF; 498 characters; selection and scoring on same OOF population | `results/tables/ablation_inner_m0_policy_alpha_0p1.json` | Completed candidate only; tau .70, min 2, max 10; not manifest-evaluated |
| E5.5 | How do locked Rungs 0–4 compare on the frozen Study D manifest? | One-shot Study D manifest; 57 runs / 17 subjects / 339 covered characters per rung | `results/tables/ablation_ladder_clean_m0.json` | Completed one-shot replay; no paired comparison survives Holm correction |

### Dead arms and superseded methods

| Dead arm | What it did | Why it is not current evidence | Record |
|---|---|---|---|
| First-token / prefix lookup | Mapped each grid key to a single token and used its next-token logit; older descriptions also claim vocabulary prefix mass marginalization | Superseded by current full-string continuation scoring with a special-key map. No validated before/after comparison exists (**NOT RUN**) | Earlier code at `3f6fe42`; replacement present at `fe044b6`; stale README sections 1.2 and 3.8 |
| Historical Study D ladder and personalization arms | Earlier EEG/LM/RAG ladder built on contaminated classifier evidence and held-out-informed stopping | Superseded by the locked clean-M0, one-shot ladder; historical uplift claims are withdrawn | `results/tables/personalization_ablation_studyd.json`; `results/tables/personalization_ablation_studyd_status.json`; `dead_arms/README.md` |
| Held-out-tuned stopping sweep | Threshold/policy selection used held-out Study D sessions | Not a valid selection procedure and replaced by clean training-only nested OOF selection | `results/tables/dynamic_stopping_threshold_sweep.json`; `docs/03_splits_and_leakage.md` |

### ITR, flashes, time, and WPM

For the ladder's Study D manifest population, each flash is assigned `2/17` seconds by the replay (`seq_seconds=2.0`, 17 row/column flashes per sequence). Character time is the sum of flash times used for that character; the reported mean time per character is the arithmetic mean over covered characters. Flashes per character is likewise the arithmetic mean of observed flashes used.

`src/evaluation/sequence_scoring.py::variable_time_itr` computes variable-time Wolpaw information transfer rate. For `N=72` Study D grid classes, accuracy `p`, and observed mean character time `T` seconds, the bit content is `log2(N) + p*log2(p) + (1-p)*log2((1-p)/(N-1))` (with zero-probability terms omitted at boundary values), floored at zero, then multiplied by `60/T` for bits/minute. `src/models/decoder.py::calculate_itr` is a wrapper that converts accuracy from percent to a fraction before calling the shared function. In the clean-M0 ladder, WPM is defined in the evaluator as `ITR / 5`; it is an ITR-derived approximation, not a separate direct character-time calculation. The `SpellerMetrics.wpm` implementation in `src/evaluation/metrics.py` instead calculates `(60 / mean_seconds_per_character) / 5`, so these two code paths should not be described as identical WPM formulas.

## Study Q, kept separate

Study Q uses typed sequences of grid keys and a different protocol from the Study D character-level text prior. In `scripts/nested_study_q_calibration_fusion.py`, the LM context starts empty for each run; after scoring a target cell, the code appends the mapped target symbol to the context. This is a teacher-forced history of the typed grid-key sequence. It is not the same contextual setting as a free-typed word and should not be pooled with Study D results. The artifact reports inner nested subject-session OOF only here (`heldout_test_data_read=false`), with clean M0 and alpha .1; prior bias is applied once before sequence evidence accumulates.

| Study Q nested OOF outcome | EEG only | LM fusion, alpha .1 |
|---|---:|---:|
| Characters / accuracy | 2,592 / 60.26% | 2,592 / 68.75% |
| Correct | 1,562 | 1,782 |
| Mean sequences | 9.74 | 9.58 |

The preceding typed-key history allows DistilGPT-2 to assign contextual probabilities among grid keys, but the artifact does not establish that the gain transfers to natural-language free spelling. Interpret it as a Study Q protocol-specific LM-prior result. The one-shot SE003 Test result is a separate population: EEG-only 55.86%, fused 59.41%, 1,296 as-recorded characters, with no stopping, flashes-per-character, or ITR claim. It is recorded in `results/tables/study_q_test_once.json` and must not be rerun.

## Component 6 handoff: RAG and personalization

```text
1. Component 5 fixes the downstream LM interface: `predict_next_char(context)` returns a grid-aligned prior; current base scoring is full-string continuation scoring.
2. The ladder uses `RAGPredictor` only at Rungs 2–4; its internals are outside this chapter.
3. Rung 2 reads `data/rag/phrase_bank_global.csv` (SHA-256 `0826a089022eefa3824e310561bba02f41d76e7bea49dadf2fe1b87a0cbacfb9`), with RAG weight .7 and adaptive weight scale 1.5.
4. Rung 3 requests subject-only retrieval from `data/rag/by_subject/phrase_bank_{subject_id}.csv`, with RAG weight 1.0.
5. Rung 4 blends global and subject retrieval, with RAG weight .7 and adaptive weight scale 1.5.
6. The rung lock sets the sufficiency gate midpoint to 15 tokens and sharpness to .2; backoff gate settings are in `splits/ablation_lock.json`.
7. Global-bank provenance and split details are locked in `splits/ablation_lock.json`; inspect Component 6 for bank construction and leakage checks.
8. The Rung 4 deferral share is verified at 312/339 covered Study D manifest characters (92.04%), from `ablation_ladder_clean_m0.json`.
9. Rungs 2–4 are bundle comparisons; Rung 2 versus Rung 1 changes RAG presence, adaptive form, and maximum weight scale together.
10. No isolated adaptive-only manifest arm exists; do not attribute Rung 2–4 outcomes to adaptive weighting or personalization alone.
```

## Reproducibility and provenance

The parser-confirmed interfaces are:

| Script | Verified argparse options | Reproduction status |
|---|---|---|
| `scripts/fit_ablation_weights.py` | `--cache`, `--scores`, `--output` | Training-OOF fit; rerun only against the corresponding clean training OOF cache and score inputs. |
| `scripts/evaluate_inner_fusion.py` | `--scores` (required), `--cache` (required), `--study`, `--alpha`, `--tau`, `--min-seq`, `--max-seq`, `--output` (required), `--policy-sweep`, `--policy-models` | Inner-OOF evaluation and policy sweep; alpha-.85 generic evaluator path. |
| `scripts/reselect_m0_policy_alpha_0p1.py` | `--output` | Fixed alpha .1 wrapper around the inner evaluator; default output is the recorded candidate artifact. |
| `scripts/run_clean_m0_ablation_ladder.py` | `--allow-heldout-eval`, `--lock`, `--manifest`, `--cache`, `--raw-dir`, `--output` | Guarded one-shot manifest evaluator already completed; **do not rerun**. |

No command is given for rerunning the one-shot ladder. The scripts' parser definitions, not an assumed CLI, are the source for these option lists.

The lock records creation commit `fe044b6`; its current locked revision is `8cb0649`; the completed ladder artifact records replay commit `e7b133c987d266eb3790b4a4a956f09c586802e5`. These commits resolve in local Git history. `git ls-remote --tags origin`, 2026-10-26, lists remote tags `ablation-final`, `baseline-d-final`, `classifier-final`, and `preproc-ablation-final`; `phase3-final`, `q-inner-final`, and `q-manifest-frozen` are local-only. A fresh command in this session failed because the environment could not resolve `github.com`; this records the dated listing supplied for the documentation, not a successful live query in this session.

## Repo reconciliation notes

* README sections 1.2 and 3.8 still describe prefix-marginalized token mass. That is contradicted by current committed full-string continuation scoring. This chapter records the code behavior; the stale README passages remain to be reconciled in a separate edit.
* The README says the prior is applied once before sequence accumulation and the Study Q artifact confirms that ordering. The ladder code also initializes sequential evidence with one initial bias.
* The README's old Study D outcomes are already labeled superseded; this chapter uses the clean-M0 locked replay only.
* Pre-existing worktree changes in `README.md`, `prototype/app.js`, and three untracked diagram files under `docs/` were left untouched. `.gitignore` was changed only to exempt this requested chapter from the repository-wide `*.md` ignore rule so Git will show it for review.
