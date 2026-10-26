# 04 — EEG classifier

**Evidence snapshot:** commit `386a78b` (`Add Study D classifier LOSO evidence`), based on `optimise` / `origin/optimise` commit `7b37a5c`. It commits the Study D LOSO runner, result JSON, score arrays, classifier documentation, and architecture diagram. The worktree also has unrelated uncommitted changes to `README.md` and `prototype/app.js`, plus ignored, untracked `INVESTIGATION_SUMMARY.md`; they are not used as evidence. Dataset/cache inputs are external or ignored unless explicitly stated. The origin branch/tag listing below is based on the user's verified check dated 2026-10-26.

## 1. Purpose, inputs, and outputs

This component maps preprocessed EEG epochs to per-flash target evidence and then to row/column or grid sequence evidence. Preprocessing and split construction are fixed inputs described in Components 2 and 3. Later language-model, fusion, retrieval, and stopping components consume classifier scores; they are not classifier training methods.

### Input cache

`scripts/build_classifier_cache.py` writes an HDF5 cache from selected training FIF files. Datasets are `X` (float32, epochs × channels × samples), `y` (uint8; 0 non-target, 1 target), `seq_idx`, `stim_code`, and `char_idx` (integer vectors), plus string vectors `subject`, `run_id`, and `condition`. Attributes include channel names/order, sample rate and time limits, source/filter/baseline settings, and manifest path/hash and run counts. If epoch metadata has `char_index` and `sequence_in_char`, these provide character and sequence indices; fixed RC fallback indexing is implemented in the builder. The builder selects training `*_Train*` files outside the held-out manifest and asserts disjointness. Study Q has a separate cache builder and frozen pool.

The final D classifier uses 1–12 Hz, baseline off, all common channels, feature window 0–0.8 s, decimation 2, and no artifact rejection. This is distinct from the 0.1–30 Hz baseline-corrected clean-reference cache used by earlier OOF experiments. Details of parser/cache preprocessing belong to [Component 2](02_preprocessing.md); run membership and leakage guards belong to [Component 3](03_splits_and_leakage.md).

### Score interface

| Stage | Interface | Evidence |
|---|---|---|
| Fit and raw epoch score | `P300Classifier.fit(X,y,groups=None)`; `decision_function(X)` returns one target-vs-nontarget score per epoch. `SklearnP300Adapter` adapts binary sklearn decision scores or log-odds from probabilities. | `src/models/classifier_interface.py` |
| Temperature calibration | `calibrate(logits,y)` fits a scalar temperature by minimizing binary log loss on the supplied scores; `calibrated_logits(X)` divides decision scores by that temperature; `calibrated_probabilities(X)` applies sigmoid. Calibration data scope is determined by the caller. | `src/models/classifier_interface.py` |
| Exported inference | `EpochFeatureScorer` accepts `(epochs,channels,times)`, checks the locked channel count/order contract, selects 0–`window_s`, decimates, flattens, applies the stored scaler and estimator, then returns logits; `calibrated_logits` divides by its stored temperature. | `src/models/classifier_interface.py`; `scripts/export_locked_clean_m0.py` |
| Row/column sequence evidence | `aggregate_flash_logits` sums calibrated flash logits by stimulus row/column code, softmaxes row and column vectors separately, and returns their outer-product grid posterior and valid-flash count. | `src/evaluation/sequence_scoring.py` |
| Group-flash sequence evidence | `aggregate_membership_logits` sums logits over each cell’s lit-mask membership and softmaxes cell scores. | `src/evaluation/sequence_scoring.py` |

The final D lock calibrates scalar temperature against row/column sequence NLL. That sequence calibration is distinct from the binary-logit calibration method available on `P300Classifier`; downstream score consumers must use the locked sequence calibration constants below.

## 2. Classifier inventory

| ID / family | Features and preprocessing | Training procedure / key settings | Code | Status | Branch/evidence |
|---|---|---|---|---|---|
| Legacy SWLDA / streaming SGD | Flattened epoch samples; historical processed FIF path. | Streaming `SGDClassifier(loss="log_loss", penalty="l2", alpha=.01, average=True, class_weight={0:1,1:5}, random_state=42)`; five passes. The old glob selected 102 runs, including 19 manifest-overlap runs. | `src/models/classifier.py`; `src/models/swlda.py` | SUPERSEDED; contaminated legacy pool | `optimise`; `classifier_protocol_audit.json` |
| M0 / clean M0 | Flattened 1–12 Hz epochs, 0–0.8 s, decim 2, all locked channels, no baseline/rejection. | Streaming SGD log-loss classifier; five passes, L2 alpha .01, averaging, class weights {0:1,1:5}. Classifier selected for the paper-facing pipeline. | `scripts/run_classical_cv.py`; `scripts/export_locked_clean_m0.py`; `src/models/classifier_interface.py` | FINAL | `optimise`; `splits/ablation_lock.json` |
| M1 tuned SWLDA | Flattened epoch samples; preprocessing-specific cache/config. | Stepwise weighted regression; enter p=.05, remove p=.10, max 25 features in the tuned classical arm (implementation supports configurable maximum). Used in matched preprocessing probes. | `src/models/swlda.py`; `scripts/run_classical_cv.py` | CANDIDATE-REJECTED for final model; used for preprocessing evidence | `optimise`; preprocessing artifacts and Component 2 |
| M2 shrinkage LDA | Flattened epoch samples; 0–0.8 s, decim 2, all channels in the matched clean-pool run. | Shrinkage LDA; maximum fit sample cap 3,000/fold in nested runs. Phase 2 screens used a 5-fold screen with 3,000 samples/fold. | `scripts/screen_shrinkage_lda.py`; `scripts/nested_sequence_calibration.py` | CANDIDATE-REJECTED | `optimise`; classifier screen/nested artifacts |
| M3a xDAWN + shrinkage LDA | xDAWN 4 filters, SCM covariance, scaler, all locked common channels and M0 feature window. | LDA `solver=lsqr`, `shrinkage=auto`, balanced priors, max fit samples 3,000, balanced sampling seed 2000. | `scripts/run_classical_cv.py`; nested evaluator | CANDIDATE-REJECTED; pre-manifest finalist | `optimise`; `splits/finalist_lock.json` |
| M3b | xDAWN + tangent-space features + logistic regression. | Three-fold nested calibration; fold temperatures recorded in artifact. | `scripts/nested_sequence_calibration.py`; `scripts/merge_nested_model_results.py` | CANDIDATE-REJECTED | `optimise`; `classifier_nested_m3b.json` |
| M2+M3a stack | Cross-fitted M2 and M3a scores. | Nested L2 logistic regression stacker, `C=.1`, balanced class weights; nested stacker cross-fit and sequence temperature. | `scripts/nested_stack_m2_m3a.py` | CANDIDATE-REJECTED | `optimise`; `classifier_stack.json` |
| M3 / M3c Riemann | xDAWN temporal-spatial covariance features. | 4 xDAWN filters; pyRiemann 0.7; balanced 3,000 fit-sample cap; logistic and minimum-distance-to-mean variants. | `scripts/run_riemann_cv.py` | CANDIDATE-REJECTED | `optimise`; `classifier_riemann_inner.json` |
| EEGNet | Epoch tensor (channels × time), configured feature window and channel selection. | BCE / softmax / combined sequence-loss modes; run-grouped CV script defaults to 30 epochs, patience 6, seed 42. No result artifact exists under `results/tables/`; the finalist lock reports MPS unavailable and CPU training not attempted. | `src/models/eegnet.py`; `scripts/run_eegnet_cv.py` | NOT RUN (recorded unavailable) | `optimise`; `splits/finalist_lock.json` |
| RP/CWT CNN | Eight-channel 2D recurrence/CWT feature path. | Exploratory CNN; historical training metadata includes test-labelled runs and is not a clean control. | `src/models/cnn_classifier.py` and feature scripts on `CNN` | DEAD / SUPERSEDED | `CNN`; `classifier_protocol_audit.json`; `INVESTIGATION_SUMMARY.md` |
| Study Q clean M0 | Study Q 9×8 group-flash data; membership masks, 1–12 Hz, no baseline, 0–0.8 s, decim 2. | Same streaming SGD family; 360 clean SE001/SE002 training runs; session-grouped OOF and separate SE003 one-shot evaluation. | `scripts/run_study_q_classifier_cv.py`; `splits/studyq_classifier_lock.json` | FINAL for Study Q locked comparison | `optimise`; Q lock/results |

## 3. Selection protocol and Phase 2 screens

Study D classifier selection used the clean RC/Train pool only: 83 runs, 17 subjects, 84,660 epochs (9,960 target; 74,700 non-target). The frozen manifest has 57 runs (19 RC/Train, 20 Dyn, 18 DynBigram). These are different populations. Component 3 documents the historical 102-run SWLDA pool and its 19-run manifest overlap. No model-selection claim below treats that legacy pool as clean evidence.

The main nested comparison used run-grouped outer OOF with inner run-grouped calibration. It reported epoch AUC, row/column sequence NLL and calibration, and character/stopping/fusion metrics. An earlier per-run classical artifact reports 249 characters; the nested/sequence artifacts report 498 characters. Those character populations are not interchangeable. The classifier-phase screen artifact used 5-fold GroupKFold by run with 3,000 fit samples and varied one configuration dimension per named screen; baseline-on unless the row says otherwise.

| Screen in `classifier_phase2_screens.json` | Varied value | Fixed values / result | Evidence |
|---|---|---|---|
| `bp_0p1_20` | Filter 0.1–20 Hz | Baseline on; window .8 s; decim 2; all channels; no artifact rejection; M2; AUC .7867 | screen JSON |
| `bp_0p5_15` | Filter 0.5–15 Hz | Same screen settings; AUC .7929 | screen JSON |
| `bp_1_12` | Filter 1–12 Hz | Same screen settings; AUC .7959 | screen JSON |
| `window_0_600` | Window 0–0.6 s | 1–12 Hz, baseline on, decim 2, all channels; AUC .8020 | screen JSON |
| `decim_3` | Decimation 3 | 1–12 Hz, baseline on, .8 s, all channels; AUC .7981 | screen JSON |
| `channels_subset` | Selected channel subset | 1–12 Hz, baseline on, .8 s, decim 2; AUC .7838 | screen JSON |
| `artifact_on` | Artifact rejection enabled, 150 µV threshold | Other M2 defaults fixed; AUC .7985 | screen JSON |
| `baseline_off` | Baseline correction off | 1–12 Hz, .8 s, decim 2, all channels, no artifact rejection; AUC .7986 | screen JSON |

These are screening AUCs, not the final nested M0-vs-M3a comparison. The screen does not itself establish causal performance effects for every configuration change. The locked M0-versus-M3a OOF and manifest evaluations below answer different questions on different populations.

## 4. Experiments and evidence

### Numerical score tables

Each table is a separate evaluation population. Percentages are character accuracy; AUC and NLL are unitless. `—` means no score is recorded in the cited artifact. Higher AUC/accuracy and lower NLL are generally preferred, but the metrics are not interchangeable.

#### Study D nested run-grouped OOF

Clean RC/Train pool; 83 runs, 17 subjects, 84,660 OOF epochs, 4,980 complete sequences, 498 characters. Fusion rows use alpha=.1 and the shared evaluator’s tau=.80/min 2/max 10 setting. Sequence NLL is calibrated row/column NLL.

| Model | Epoch AUC | EEG-only char. acc. | Fused char. acc. | Row/column NLL | Evidence |
|---|---:|---:|---:|---:|---|
| M0 clean SGD | .8028 | 79.52% | 81.93% | 1.6281 | `classifier_nested_calibration.json`; README at `494066a` |
| M2 shrinkage LDA | .7954 | 78.31% | 80.32% | 1.6207 | `classifier_nested_calibration.json`; README at `494066a` |
| M3a xDAWN + shrinkage LDA | .8155 | 80.12% | 82.53% | 1.5161 | `classifier_nested_calibration.json`; README at `494066a` |
| M3b xDAWN + tangent + LR | .8030 | 76.31% | 79.32% | 1.5580 | `classifier_nested_m3b.json`; README at `494066a` |
| M2 + M3a stack | .8243 | 80.52% | 82.13% | 1.4674 | `classifier_stack.json`; README at `494066a` |

#### Study D superseded exploratory per-run evaluator

These early standalone evaluator artifacts report 84,660 epochs but only 249 characters. The discrepancy is due to the earlier evaluator's sequence/character grouping path: the fixed sequence-indexing audit found fixed-RC characters with ten sequences, and cache/nested sequence grouping was corrected before the finalist lock and nested comparison (`splits/finalist_lock.json`, `e16ed22`). Therefore these 249-character character metrics are **superseded exploratory outputs**, not a separate evaluation population and not evidence for final model selection. The table is retained for traceability; use the 498-character nested table and the clean M0 LOSO table for current cross-validated results.

| Model | Epoch AUC | Full-repetition char. acc. | Locked-stopping char. acc. | Evidence |
|---|---:|---:|---:|---|
| M0 clean SGD | .8027 | 32.13% | 24.10% | `classifier_classical_inner.json`; run metadata commit `53837ad` |
| M1 tuned SWLDA | .7599 | 26.91% | 25.70% | `classifier_classical_inner.json`; run metadata commit `53837ad` |
| M3a xDAWN + LDA (classical artifact) | .8129 | 34.54% | 30.52% | `classifier_riemann_inner.json` |
| M3 xDAWN tangent-space LR | .8032 | 31.73% | 31.73% | `classifier_riemann_inner.json` |
| M3c Riemannian MDM | .7936 | 32.53% | 32.93% | `classifier_riemann_inner.json` |

#### Study D one-shot manifest

Frozen manifest; 57 runs, 17 subjects, 338 characters including 11 zero-coverage. These values are not OOF scores.

| Model | EEG-only char. acc. | Fused char. acc. | Mean sequences | Variable-time ITR (bits/min) | Evidence |
|---|---:|---:|---:|---:|---|
| M0 clean SGD | 72.49% | 78.40% | 3.82 | 32.09 | `classifier_manifest_eval.json` |
| M3a xDAWN + shrinkage LDA | 60.95% | 74.26% | 2.95 | 38.25 | `classifier_manifest_eval.json` |
| Legacy contaminated SWLDA artifact, replayed under clean M0 policy | — | 68.93% | — | — | `classifier_manifest_eval.json`; audit only, not historical pipeline reproduction |

M3a−M0 fused accuracy was −4.20 percentage points (subject-bootstrap 95% CI −10.25 to +1.08; Wilcoxon p=.2662, Holm p=.2662). This paired estimate is for the manifest population above.

#### Study D leave-one-subject-out (LOSO)

This is a separate, newly generated, **EEG-only** evaluation of clean M0; no LM prior or fusion alpha is applied. Each outer fold holds out one entire subject; calibration temperature is fitted using inner run-grouped OOF scores from the other 16 subjects only. The evaluated data are the clean RC/Train cache, not the frozen manifest. This gives a subject-generalization estimate and is not the manifest’s Dyn/DynBigram condition-shift evaluation.

| Model / score mode | Outer validation | Epoch AUC | Calibrated row/column NLL | Full-repetition EEG-only char. acc. | Stopped EEG-only char. acc. | Mean sequences used |
|---|---|---:|---:|---:|---:|---:|
| Clean M0 streaming SGD, EEG-only | 17 held-out subjects; 83 RC/Train runs / 84,660 epochs / 498 characters total | .7869 | 1.6639 | 74.70% | 73.90% | 6.86 |

Evidence: `results/tables/study_d_classifier_loso.json` and foldwise OOF logits in `results/tables/study_d_classifier_loso_scores.npz`; runner: `scripts/run_study_d_classifier_loso.py`. The run used `tau=.80`, min 2/max 10; 17 fold temperatures range from 4.3012 to 4.9404. Cache SHA-256 is `7615c89e80b266d70403180d9f3880611d8201dd3c7e78384c912212b4f3093e`; manifest SHA-256 is `de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b`. The runner records `manifest_test_runs_loaded=false`. Against the 498-character run-grouped OOF EEG-only result at the same tau/min/max (79.52%), LOSO stopped accuracy is lower by 5.56 percentage points (73.90%). This is a descriptive cross-protocol comparison, not an independent paired significance test. This LOSO result estimates performance across held-out training subjects; do not substitute it for the one-shot frozen-manifest result or pool them.

#### Study Q (separate dataset and split)

| Protocol / model | Population | Epoch AUC | Character accuracy | Evidence |
|---|---|---:|---:|---|
| Session-grouped OOF, clean M0 | 360 train runs, 36 subjects; 2,592 characters | .746 | 57.37% full available-sequence | `study_q_classifier_inner.json`; README at `494066a` |
| LOSO, clean M0 | 2,592 characters; leave-one-subject-out folds | — | 60.69% full available-sequence | `study_q_classifier_loso.json` |
| SE003 one-shot, EEG-only M0 | 180 held-out runs, 36 subjects, 1,296 as-recorded characters | — | 55.86% | `study_q_test_once.json` |
| SE003 one-shot, fused M0 | Same 180-run / 1,296-character test population | — | 59.41% | `study_q_test_once.json` |

No EEGNet result is recorded. The old 32.66% SWLDA number is in the historical README table but is contaminated and held-out-informed. The 22.06% CNN accuracy and .90 within-group AUC are only in ignored, untracked `INVESTIGATION_SUMMARY.md`, with no population label; they are excluded from these score tables.

| Experiment / question | Method | Result with population | Evidence / commit | Status |
|---|---|---|---|---|
| Legacy SWLDA baseline and leakage audit | Historical training glob compared with frozen D manifest. | 102 old RC/Train runs; 19 overlap manifest runs; clean pool 83 runs. CNN metadata had 213 training runs, 143 Test-labelled, zero manifest overlap; still unsuitable as a clean control because it contains Test-labelled runs. | `classifier_protocol_audit.json`; `62ce804` | Legacy results superseded; CNN control invalid pending clean retraining |
| Classical M0/M1/M2/M3a OOF | Three-fold GroupKFold by run, clean cache, 3,000 fit samples/fold. | OOF per-run artifact: M0 AUC .8027; M1/M2/M3a values are in `classifier_classical_inner.json`. Character metrics there cover 249 characters, not the 498-character nested population. | `classifier_classical_inner.json`; run metadata names commit `53837ad` and says `git_dirty=true` | Exploratory screen |
| Phase 2 shrinkage screens | 5-fold run-grouped M2 screens; varied filter, baseline, channels, decimation, window, artifact rejection. | AUCs .7838–.8020 on the screen’s OOF epoch population; each row above identifies its variant. | `classifier_phase2_screens.json`; `53837ad` | Screen, not final selection |
| Nested model/calibration comparison | Three outer and three inner run-grouped folds; calibration learned only from inner OOF within outer training pools. | 83 D runs / 17 subjects / 84,660 OOF epochs / 4,980 complete sequences. Epoch AUC: M0 .8028; M2 .7954; M3a .8155; M3b .8030. Calibrated row/column NLL: M0 1.6281; M2 1.6207; M3a 1.5161; M3b 1.5580. | `classifier_nested_calibration.json`, `classifier_nested_m3b.json`; `88d19ec` | Candidate comparison |
| M2+M3a score stack | Nested cross-fit stacker plus sequence calibration. | Same nested D population: 4,980 sequences; calibrated row/column NLL 1.4674. The lock’s pre-set adoption comparison says fused stopped accuracy was below M3a by .0028039 despite lower sequence NLL, so stack not selected. | `classifier_stack.json`; `splits/finalist_lock.json` | Candidate-rejected |
| Riemannian variants | Run-grouped 3-fold CV, 3,000 balanced fit samples, 1–12 Hz baseline-off cache. | 84,660 epochs / 83 runs. M3 xDAWN tangent-space logistic AUC .8032; M3c Riemannian MDM AUC .7936. The artifact’s character population is 249. | `classifier_riemann_inner.json` | Candidate-rejected |
| Finalist lock | Freeze M0 baseline and M3a candidate before the manifest comparison. | Lock specifies M0 sequence temperature 4.1869360663 and M3a 1.2981389855; preprocessing 1–12 Hz, baseline off, window 0–.8 s, decim 2, all common channels, 83 training runs. | `splits/finalist_lock.json`; parent lock hash is recorded inside `ablation_lock.json` | Pre-evaluation lock; M3a not final |
| M3a vs M0 clean nested OOF | Nested run-grouped OOF, sequence calibration, fixed alpha=.1 stopping/fusion reporting in README parity table. | 498 characters / 83 runs / 17 subjects: epoch AUC M0 .8028 vs M3a .8155. At alpha=.1, tau=.80, min 2/max 10, EEG-only/fused accuracy M0 79.52/81.93%; M3a 80.12/82.53%. Difference in fused character accuracy +0.44 pp (95% CI −3.12 to +4.22; Holm p=1.0). M3a row/column NLL lower by .107 (CI −.161 to −.048; Holm p=.065). | README committed at `494066a`; `phase0_oof_evaluator_parity.json`; nested artifacts | OOF candidate evidence; not held-out |
| Full nested classifier screen | Same clean D nested run-grouped OOF population: 498 characters / 83 runs / 17 subjects; alpha=.1, tau=.80, min 2/max 10. | EEG-only/fused character accuracy: M0 79.52/81.93%; M2 78.31/80.32%; M3a 80.12/82.53%; M3b 76.31/79.32%; M2+M3a stack 80.52/82.13%. Epoch AUC: M0 .8028, M2 .7954, M3a .8155, M3b .8030, stack .8243. | README committed at `494066a`; nested and fusion result tables | Mixed metrics; paired accuracy evidence did not establish a clear winner |
| M3a vs M0 one-shot manifest | Frozen held-out manifest evaluation; not a rerun command. | 57 runs / 17 subjects; 338 characters including 11 zero-coverage. Fused/EEG-only M0 78.40/72.49%; M3a 74.26/60.95%. M3a−M0 fused delta −4.20 pp (subject-bootstrap 95% CI −10.25 to +1.08; Wilcoxon/Holm p=.2662). M3a used 2.95 vs M0 3.82 mean sequences and had variable-time ITR 38.25 vs 32.09 bits/min. | `classifier_manifest_eval.json`, run commit `e16ed2236e41480a7eb494a233d7a1e18c94ef16`; `splits/finalist_lock.json` | Completed one-shot; M3a rejected |
| Alpha=.1 clean M0 stopping-policy candidate | Re-select policy on clean nested OOF at manifest alpha. | 498 clean OOF characters. Candidate tau=.70, min 2, max 10; same 81.93% accuracy as tau=.80; mean sequences 5.73 vs 6.45 (as recorded in Component 3). This candidate was not evaluated on manifest; manifest lock remains tau=.80. | `ablation_inner_m0_policy_alpha_0p1.json`; `docs/03_splits_and_leakage.md`; commit `2926644` and doc commit `7b37a5c` | Candidate only; not a classifier change |
| Study Q clean classifier | Session-grouped OOF on clean Study Q training and separate LOSO analysis. | Session-grouped: 360 train runs, 36 subjects, 2,592 characters, epoch AUC .746 in README; full available-sequence accuracy .5737 in `study_q_classifier_inner.json`. LOSO is a separate protocol and artifact; do not combine its metrics with session-held-out results. | `study_q_classifier_inner.json`, `study_q_classifier_loso.json`, `study_q_classifier_by_condition.json`, `studyq_classifier_lock.json` | Study Q evidence; separate population |
| Study Q one-shot SE003 | Frozen session-held-out test-once comparison, as-recorded character depth. | 180 held-out runs / 36 subjects / 1,296 as-recorded characters. Clean M0 EEG-only 55.86%; fused 59.41%. No stopping, flashes/character, or ITR claim under this Q lock. | `study_q_test_once.json`; `studyq_classifier_lock.json`; `splits/study_q_manifest.json` | Completed one-shot |

### Population reconciliation

The nested clean OOF screen favors M3a numerically over M0 on fused accuracy, while the one-shot manifest point estimate favors M0. These results must not be merged or described as one estimate. The training pool is RC/Train only, while the manifest additionally contains Dyn and DynBigram. The sign change is consistent with sensitivity to condition shift; it does not prove that condition shift caused the difference. M3a therefore remained a candidate and was not selected after the manifest comparison.

## 5. Calibration and score constants

| Model / scope | Calibration fit | Value/evidence |
|---|---|---|
| M0 final D scorer | Scalar temperature selected to minimize row/column sequence NLL on full clean-pool run-grouped OOF scores. | `4.1869360663`; `splits/finalist_lock.json`, repeated in `splits/ablation_lock.json`. |
| M3a finalist comparison | Same clean-pool run-grouped OOF sequence-NLL scope. | `1.2981389855`; `splits/finalist_lock.json`; candidate only. |
| Nested D comparison | Temperature fit on inner run-grouped OOF logits within each outer training fold; each model has fold-specific values. | M0 `[6.1522,6.5459,5.9135]`; M2 `[1.7889,1.8304,1.8274]`; M3a `[1.3988,1.3763,1.4018]`; M3b `[1.2945,1.3117,1.3114]`. `classifier_nested_calibration.json` / `classifier_nested_m3b.json`. |
| Study Q M0 | Scalar temperature fitted to clean training-only OOF membership-grid NLL. | `1.6921645662`; `splits/studyq_classifier_lock.json`. |

The `classifier_m2_calibration.json` and `classifier_m3a_calibration.json` artifacts also fit a scalar binary-logit temperature on all run-grouped OOF training-pool logits. Their temperatures are 1.6024559894 and 1.2416938364, respectively. These earlier calibration artifacts are not the nested fold-specific sequence calibration and are not the final M0 constant. The project note that Platt scaling was tried on CNN row/column decoding and replaced by within-row/column softmax has no matching `Platt` record in the searched code/docs/results or CNN branch; report it as **UNVERIFIED project note**, not an established experiment.

## 6. Dead arms and superseded results

| Arm / claim | Evidence and disposition |
|---|---|
| Old SWLDA/SGD comparison | `classifier_protocol_audit.json` finds 19 of 102 historical training runs overlap the held-out manifest. The clean pool has 83 runs. In the later clean same-manifest audit, the old contaminated SWLDA artifact scored 68.9% fused vs clean M0 78.4% on 57 manifest runs / 338 characters (11 zero-coverage); this is a new-policy audit, not a reconstruction of the old pipeline or a causal estimate of overlap bias. Historical scores remain invalid clean held-out estimates. |
| CNN RP/CWT arm | `origin/CNN` ends at `911c89a`; its branch audit records 213 CNN training runs, 143 Test-labelled, and 0 manifest-overlap. Retrain on clean RC/Train before using it as a control. A committed historical README table gives the old SWLDA baseline as 32.66% character accuracy on its held-out test split, later identified as contaminated/held-out-informed; it is not a valid clean comparison (README at `494066a`, historical §6.1). The 22.06% CNN accuracy and .90 within-group AUC appear only in ignored, untracked `INVESTIGATION_SUMMARY.md`, with no run/subject/character population or branch provenance; they are **worktree-only, UNVERIFIED**, not branch evidence. Commit `38d47fb` (“dead end”) exists on the local-only `CNN` branch and is absent from origin. |
| M1 tuned SWLDA | Earlier 249-character classical run-grouped OOF: AUC .7599 and full-repetition accuracy 26.91%, vs M0 .8027 and 32.13% on the same artifact population/configuration. It remained a preprocessing probe, not final. This is not the 498-character nested comparison. Evidence: `classifier_classical_inner.json`. |
| M2 shrinkage LDA | On 498-character nested clean OOF at alpha=.1, M2 fused accuracy was 80.32% vs M0 81.93%; AUC .7954 vs .8028. Row/column NLL was slightly lower (1.6207 vs 1.6281), so evidence is mixed rather than uniformly worse. No manifest M2 comparison is reported. Evidence: `classifier_nested_calibration.json`; README at `494066a`. |
| M3a xDAWN + shrinkage LDA | On nested clean OOF (498 characters), fused accuracy was 82.53% vs M0 81.93%, with no significant paired accuracy advantage; on the separate 57-run manifest, fused accuracy was 74.26% vs M0 78.40% (paired delta −4.20 pp, CI includes zero). The manifest result and locked primary metric led to retaining M0; the OOF/manifest sign reversal is explicit in §4. Evidence: `classifier_nested_calibration.json`; `classifier_manifest_eval.json`. |
| M3b xDAWN + tangent + LR | On nested clean OOF (498 characters), fused accuracy was 79.32% vs M0 81.93%; epoch AUC was .8030 vs .8028 and calibrated NLL 1.5580 vs 1.6281. Better NLL/AUC did not translate to higher stopped character accuracy; no manifest result is present. Evidence: README at `494066a`; `classifier_nested_m3b.json`. |
| M2+M3a stack | On nested clean OOF (498 characters), fused accuracy was 82.13% vs M3a 82.53%; epoch AUC .8243 and sequence NLL 1.4674. The pre-set lock rule rejected it because fused stopped accuracy was 0.28 pp below M3a despite lower sequence NLL. No manifest result is present. Evidence: README at `494066a`; `classifier_stack.json`; `splits/finalist_lock.json`. |
| Riemannian models | Earlier 249-character run-grouped OOF: xDAWN tangent-space LR AUC .8032 and full-repetition accuracy 31.73%; MDM AUC .7936 and accuracy 32.53%; the Riemann artifact’s xDAWN-LDA row is .8129 AUC / 34.54% full-repetition accuracy. The similarly configured M0 classical artifact had AUC .8027 and full-repetition accuracy 32.13%; exploratory only, not a locked nested selection or manifest comparison. Evidence: `classifier_riemann_inner.json`; `classifier_classical_inner.json`. |
| Quarantined 87.15% fused M0 figure | Not reproducible under the corrected shared evaluator; recomputed clean OOF is 81.93% for M0 fused at alpha=.1, tau=.80 on 498 characters. Do not quote 87.15% as a current score. | `phase0_oof_evaluator_parity.json`; `splits/baseline_d_lock.json`; committed README at `494066a`. |
| EEGNet | Lock says MPS unavailable and CPU training was not attempted; no EEGNet result JSON in `results/tables/`. No performance number exists, so mark **NOT RUN**, not a negative result. |
| Platt scaling note | No occurrence found in the checked repository text and CNN history. **UNVERIFIED**. |
| Cross-branch refs and origin tags | As checked by the user on origin, 2026-10-26: origin has tags `ablation-final`, `baseline-d-final`, `classifier-final`, `preproc-ablation-final`; `phase3-final`, `q-inner-final`, `q-manifest-frozen` are not on origin. `origin/CNN` ends at `911c89a`; local-only commit `38d47fb` is absent from origin. Other local ref observations: `7b37a5c` is `optimise`/`origin/optimise` HEAD; `SKB-J8` adds initial CNN implementation at `5889290`; `origin/codex/create-detailed-project-documentation` adds `a142243` (README expansion only); `main` and `codex/personalized-rag-focus-ui` have zero commits absent from `optimise`. |

## 7. Final locked classifier

| Item | Frozen value |
|---|---|
| Paper-facing Study D model | `clean_m0`, streaming SGD epoch scorer |
| Features | 1–12 Hz; epoch −0.1…0.8 s at parse; baseline off; no artifact rejection; all common channels; classifier feature window 0…0.8 s; decim 2 |
| Estimator | `SGDClassifier(loss="log_loss", penalty="l2", alpha=.01, average=True, class_weight={0:1,1:5})`; five streaming passes in the lock description |
| Calibration | Row/column sequence-NLL scalar temperature `4.1869360663`, fit on clean-pool run-grouped OOF scores |
| Stopping value supplied to downstream policy | Manifest lock: tau=.80, min 2, max 10. Component 3 records a separate alpha=.1 tau=.70 OOF candidate, not a replacement lock. |
| Lock files | `splits/finalist_lock.json` SHA-256 `46e7c2008f89c772158e3f11aedb4af799e09b1fbc30a0c628df6c079731509a`; `splits/ablation_lock.json` SHA-256 `840c12db474735757ece08ae70b489795870c09933dfd18819a34a32de895095`; `splits/heldout_manifest.json` SHA-256 `de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b`. Study Q lock hashes are in §9. |
| Scorer export | `scripts/export_locked_clean_m0.py` exports `data/processed/clean_m0_epoch_scorer.pkl` from `data/cache/classifier_1_12_no_baseline.h5` using `configs/classifier_optimization/preprocessing/baseline_off.json`. |
| Byte-level scorer provenance | **UNVERIFIED**: scorer SHA is not recorded in the lock or ladder JSON. The historical manifest result also lacks its training-cache SHA. |

The repository requires scikit-learn `<1.6` because MNE 1.8 estimator tags require it (`requirements.txt`). The documented constraint is reproducibility guidance; no earlier scikit-learn failure is asserted here.

## 8. Chronology and selection interpretation

| Evidence point | Change / finding |
|---|---|
| Historical classifier path | Legacy SWLDA/streaming SGD used a 102-run pool, 19 runs overlapping the later manifest. `classifier_protocol_audit.json` establishes this conflict. |
| Clean classifier protocol | `classifier-final` (`dbf5293`) records clean-pool classifier evaluation. The clean pool uses 83 RC/Train runs; held-out runs are excluded. |
| Nested calibration and finalist | `8512d46` locks M3a as the sole pre-evaluation candidate and M0 as baseline. The lock defines sequence-level calibration from clean run-grouped OOF. |
| Manifest evaluation | `e16ed22` records the corrected fixed sequence grouping before one-shot manifest replay. M3a’s manifest point estimate is worse than M0 despite its stronger OOF score/NLL evidence. |
| Shared OOF and final selection | `88d19ec` consolidates evaluation; the OOF parity audit identifies the old 87.15% M0 figure as unreproducible. Paper-facing lock `ablation_lock.json` selects clean M0. |
| Alpha=.1 policy candidate | `2926644` records clean OOF re-selection; `7b37a5c` documents it in Component 3. Candidate tau=.70 is not manifest-evaluated and does not change the classifier lock. |

The condition-shift interpretation is limited: nested OOF is drawn from RC/Train, while the one-shot manifest includes Dyn and DynBigram. The observed M3a-minus-M0 sign reversal motivates caution in selecting from RC/Train alone; it does not isolate a causal condition effect. The pre-registered M0 baseline is retained.

## 9. Study Q classifier

Study Q uses another dataset and group-flash design. It has 9×8 grid membership masks, not the Study D row/column flash scoring setup; it also has a distinct session-held-out split and its own filter/cache/calibration lock. Do not compare Q metrics numerically with D metrics as if they shared a population.

| Study Q analysis | Population and result | Evidence |
|---|---|---|
| Clean training | 360 SE001/SE002 runs from 36 subjects; same-session SE003 Train runs excluded. | `splits/studyq_classifier_lock.json`; `splits/study_q_excluded_training_runs.json` |
| Session-grouped OOF | 2,592 characters; README reports epoch AUC .746. Inner artifact reports full available-sequence character accuracy .5737. | `study_q_classifier_inner.json`; `README.md` at `494066a` |
| LOSO | Separate leave-one-subject-out analysis; full available-sequence accuracy .6069 across 2,592 characters. This is not session-held-out CV. | `study_q_classifier_loso.json` |
| SE003 one-shot | 180 runs, 36 subjects, 1,296 as-recorded characters; M0 EEG-only 55.86%, fused 59.41%. No stopping, flashes-per-character, or ITR estimand in the lock. | `study_q_test_once.json`; `splits/study_q_manifest.json`; `splits/studyq_classifier_lock.json` |
| Calibration | Membership-grid sequence NLL, clean training-only OOF; temperature 1.6921645662. | `splits/studyq_classifier_lock.json` |

## 10. Limitations and open items

- D OOF and manifest results have different populations and reverse the M3a-vs-M0 point-estimate sign; the cause is not isolated.
- Manifest evaluator has 338 characters including 11 zero-coverage; the clean ablation ladder reports 339. The one-character population discrepancy is **UNVERIFIED**; preserve each artifact’s population label.
- The manifest artifact does not record its training-cache SHA. Serialized M0 scorer byte provenance is **UNVERIFIED**.
- The CNN branch contains Test-labelled training metadata; it is not a clean control without retraining.
- Study D clean M0 LOSO is EEG-only. Its stopped character accuracy is 73.90% versus 79.52% for run-grouped OOF EEG-only at the same stopping policy (−5.56 pp); this is descriptive across protocols, not a paired significance test. Study Q's LOSO score (60.69%) is from a different dataset and remains separate.
- EEGNet is **NOT RUN** under the recorded environment constraint. No EEGNet score is available.
- The Platt-scaling story is an **UNVERIFIED** project note.
- Origin tags and the CNN remote branch tip are recorded from the user's origin check dated 2026-10-26; the local environment could not independently reach GitHub during this turn.
- M2/M3a binary-logit calibration artifacts, nested sequence temperatures, and final lock temperatures use different scopes. Do not substitute one for another.
- The earlier classical/Riemann artifacts contain 249-character metrics from a superseded sequence/character grouping path. The corrected nested results cover 498 characters. Treat the 249-character table as exploratory and superseded; do not interpret the accuracy gap as a separate subject population effect.

## 11. Reproduction and provenance checks

Commands below are limited to parser-confirmed arguments. Cache and dataset inputs are external. One-shot held-out evaluators have already been run; do not rerun them for routine reproduction.

| Purpose | Command | Note |
|---|---|---|
| Classical run-grouped CV | `.venv/bin/python scripts/run_classical_cv.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --folds 3 --max-fit-samples 3000 --output results/tables/classifier_classical_inner.json` | Parser exposes these flags; script also has `--scores-output` and `--models`. |
| Study D clean M0 LOSO | `.venv/bin/python scripts/run_study_d_classifier_loso.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --inner-folds 3 --tau 0.80 --min-sequences 2 --max-sequences 10 --output results/tables/study_d_classifier_loso.json --scores results/tables/study_d_classifier_loso_scores.npz` | Parser-confirmed. Outer split is subject LOSO; calibration is nested within the other subjects. EEG-only; clean training cache only. |
| Shrinkage screens | `.venv/bin/python scripts/screen_shrinkage_lda.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --folds 5 --max-fit-samples 3000 --output results/tables/classifier_phase2_screens.json` | Screen uses the CLI default/config; specific historical variants require their corresponding caches/configs. |
| Riemann CV | `.venv/bin/python scripts/run_riemann_cv.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --folds 3 --fit-samples 3000 --output results/tables/classifier_riemann_inner.json` | Validated parser flags; model choices and scores-output are optional. |
| Nested calibration | `.venv/bin/python scripts/nested_sequence_calibration.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --models M0 M2 M3a M3b --outer-folds 3 --inner-folds 3 --max-fit-samples 3000 --output results/tables/classifier_nested_calibration.json --scores results/tables/classifier_nested_calibration_scores.npz` | Run only if regenerating training-only OOF artifacts; script has no held-out evaluation. |
| Study Q session-grouped CV | `.venv/bin/python scripts/run_study_q_classifier_cv.py --scheme session --folds 3 --output results/tables/study_q_classifier_inner.json` | Parser also accepts cache, manifest, exclusions, scores. Uses Study Q paths/defaults. |
| Lock digests | `shasum -a 256 splits/finalist_lock.json splits/ablation_lock.json splits/heldout_manifest.json splits/studyq_classifier_lock.json splits/study_q_manifest.json` | Compare to hashes embedded in lock/result files where present. |
| Inspect one-shot D evaluator CLI | `scripts/evaluate_classifier_manifest.py` | Parser includes `--allow-heldout-eval`, `--dry-run-training`, `--lock`, `--manifest`, `--cache`, `--config`, `--raw-dir`, `--old-model`, `--output`. Already run; do not rerun. |

No command is given for an EEGNet result because no result artifact is present; no CPU training was recorded. Calibration, export, and merger scripts have their own parser options; consult their argparse definitions before invoking them.

## 12. Handoff to Component 5 — LM prior and fusion

1. Final Study D epoch scorer: clean M0, streaming SGD, locked 1–12 Hz, baseline off, 0–0.8 s features, decim 2, all common channels.
2. M0 sequence temperature: `4.1869360663`, clean-pool run-grouped OOF row/column NLL fit (`splits/ablation_lock.json`).
3. Downstream interface: calibrated per-flash logits → sum by row/column → separate softmax vectors → outer-product grid posterior.
4. Manifest stopping lock: tau=.80, min 2, max 10; alpha=.1 clean OOF candidate tau=.70 was not manifest-evaluated.
5. D nested OOF population: 83 RC/Train runs, 17 subjects, 84,660 epochs, 4,980 complete sequences, 498 characters.
6. D nested OOF M0/M3a epoch AUC: .8028/.8155; at alpha=.1 fused character accuracy 81.93%/82.53% on 498 characters.
7. D manifest population: 57 runs, 17 subjects, 338 characters (11 zero-coverage); M0 fused/EEG-only 78.40%/72.49%.
8. D manifest M3a fused 74.26%; M3a−M0 −4.20 pp (subject-bootstrap 95% CI −10.25 to +1.08); M3a was rejected.
9. Study Q is separate: 360 clean training runs; SE003 Test 180 runs/36 subjects/1,296 as-recorded characters; M0 EEG-only/fused 55.86%/59.41%; Q temperature 1.6921645662.
10. Study D clean M0 LOSO is EEG-only (no LM/fusion): 17 subjects held out one at a time; 498 characters; AUC .7869; calibrated row/column NLL 1.6639; full/stopped accuracy 74.70/73.90%; 6.86 mean sequences.
11. LOSO stopped EEG-only accuracy 73.90% vs run-grouped OOF EEG-only 79.52% at the same policy: −5.56 pp, descriptive cross-protocol gap.
12. Neither D OOF, D manifest, D LOSO, nor Q numbers should be pooled across studies or evaluation populations; all future priors consume the locked calibrated score interface.
