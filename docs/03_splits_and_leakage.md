# 03 — Splits, held-out manifests, and leakage audit

**Evidence snapshot:** split, classifier, and leakage baseline at `62ce804`; alpha 0.1 selection helper and result artifact at `2926644`. Component 2 documentation at `b9e9266` (`preproc-ablation-final`). `phase3-final` is local-only. The current worktree has unrelated uncommitted edits to `README.md` and `prototype/app.js`; they are not used as evidence here. Dataset inputs under `data/` are external/ignored. Hashes below are SHA-256 of the named committed files.

## 1. Purpose and pipeline position

This component defines the run-level partitions that constrain preprocessing caches, classifier fitting, internal validation, and final evaluation. It records how the Study D overlap was detected, how Study Q's session-held-out split is frozen, and what evidence can and cannot support later component claims. Preprocessing behavior is described by Component 2; this document references it only where it enforces a split.

## 2. Data partitions

| Study / pool | Conditions | Runs | Subjects | Epochs / characters | Role | Defining evidence |
|---|---|---:|---:|---|---|---|
| D clean training | RC/Train only | 83 | 17 | 9,960 target + 74,700 non-target epochs (84,660 total) | Classifier fitting, run-grouped OOF, training-only language/weight fitting | `results/tables/classifier_protocol_audit.json`; `splits/heldout_manifest.json`; `62ce804` |
| D frozen manifest | RC/Train 19; Dyn 20; DynBigram 18 | 57 | 17 | Manifest classifier replay: 338 total characters, of which 11 are terminal zero-coverage characters; ladder artifact reports 339 characters per rung | One-shot held-out evaluation, condition-stratified reporting | `splits/heldout_manifest.json` (SHA-256 `de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b`); `results/tables/classifier_manifest_eval.json`; `results/tables/ablation_ladder_clean_m0.json` |
| D historical SWLDA pool | RC/Train | 102 | 17 | Not reported as a comparable held-out population | Contaminated legacy training pool; 19 run IDs overlap the D manifest | `results/tables/classifier_protocol_audit.json`; `62ce804` |
| Q clean training | SE001/SE002 Train, with same-session SE003 Train excluded | 360 selected training runs | 36 | Training epochs reported in Q audit/cache artifacts; no single character total asserted here | Clean classifier fitting and session-grouped OOF | `splits/study_q_excluded_training_runs.json`; `results/tables/study_q_audit.json`; `splits/studyq_classifier_lock.json`; `f5010b0` |
| Q frozen manifest | SE003 Test; ColorIntensification 55, Grey-to-Color 60, Grey-to-White 65 | 180 | 36 | 1,296 as-recorded characters | External, one-shot evaluation after inner analysis freeze | `splits/study_q_manifest.json` (SHA-256 `39ed6801b2545f37f5f0b1019c01bec0e8fd6b7a6b4418c7014c00b9c689c973`); `results/tables/study_q_audit.json`; `results/tables/study_q_phase1_gate.json`; `f5010b0` |

The Study D clean-pool epoch totals come from the protocol audit's per-subject counts. The manifest's character totals differ by evaluator and artifact; see §5. The Study Q audit describes a path-derived frozen split, and its warning says no Q EEG or labels had been evaluated at the time of that audit; later evaluation is separately recorded in the test-once artifacts.

## 3. Split construction and enforcement

| Mechanism | Grouping / rule | Enforcement evidence |
|---|---|---|
| Study D manifest | Subject-keyed lists of run IDs; session/run is the partition unit. SHA-256 is pinned in the manifest sidecar, classifier locks, and evaluation output. | `scripts/audit_classifier_protocol.py:main` computes the digest, counts conditions, and calls `assert_training_pool_disjoint`; `src/evaluation/classifier_protocol.py:assert_training_pool_disjoint` is the shared assertion. |
| Study D cache | Select only `Train` epoch files whose run ID is absent from the held-out manifest; assert selected training paths are disjoint. Cache stores manifest path, SHA-256, and held-out run count. | `scripts/build_classifier_cache.py:build_cache`; parser options in `main`: `--epochs`, `--manifest`, `--output`, `--source`, `--raw-dir`, `--no-baseline-correction`, `--l-freq`, `--h-freq` (and the remaining cache options in that parser). |
| Preprocessing registry | For Study D, expose labels to training only when the run is non-Test, is not in the manifest, and is in an allowed RC/Train condition. The code writes a separate evaluation registry with mode `0600`. | `src/data/batch_preprocess.py:main`, especially held-out ID load and `keep_training_label` selection. |
| Study D label vault | Registry partition keeps only non-held-out Study D RC/Train labels in the training-visible registry; protected Study D labels and other studies remain in the vault. Files are created with restrictive permissions by the vault workflow. | `scripts/separate_classifier_label_vault.py:partition_registry`, `split_registry`; only CLI mutation switch is `--apply`. |
| Study Q manifest | Inventory paths by subject/session/split/condition; SE003 Test IDs become the manifest. SE003 Train IDs are separately listed as excluded training runs. Frozen inventory must match unless `--refreeze` is explicitly supplied before evaluation. | `scripts/audit_study_q.py:audit`; `splits/study_q_manifest.json`; `splits/study_q_excluded_training_runs.json`; both digests are recorded in audit output. |
| Study Q cache | Selected Q training runs are checked disjoint from manifest IDs and carry the manifest digest and held-out session metadata. | `scripts/build_study_q_classifier_cache.py:build`; parser options `--output`, `--max-runs`; `src/evaluation/classifier_protocol.py:assert_training_pool_disjoint`. |
| Study Q label vault | Assert clean training IDs do not intersect held-out or excluded IDs; verify vault coverage and preserve test labels in the vault. | `scripts/separate_study_q_label_vault.py` (including `split_registry`); only CLI mutation switch is `--apply`. |
| Phrase bank / fusion weights | `scripts/build_phrase_bank_from_registry.py:is_test` assigns sessions by `SHA256("phrase-bank-v1:" + session_id) % 100 < 20`. Training phrase banks omit that 20% validation subset. The ablation lock uses the corresponding 20% salted SHA-256 training-run OOF split for fusion weight fitting; the RAG bank excludes those runs and all manifest runs. | `scripts/build_phrase_bank_from_registry.py:is_test`; `scripts/fit_ablation_weights.py`; `splits/ablation_lock.json` fields `weight_fit_scope`, `OOF_validation_runs`, `bank_built_from`. |

The one-shot manifest evaluators verify manifest hashes and require an explicit held-out access flag. The Study Q evaluator creates an exclusive `study_q_test_once.started` marker before reading labels. `--allow-heldout-eval` is present in `scripts/evaluate_classifier_manifest.py`, `scripts/run_clean_m0_ablation_ladder.py`, `scripts/run_personalization_ablations.py`, `scripts/evaluate_study_q_test_once.py`, and `prototype/server.py`; it is not a general training permission. The evaluator source checks and lock hashes define the authorized evaluation boundary.

## 4. Leakage audit chronology

| Event | Finding / consequence | Evidence |
|---|---|---|
| Historical Study D evaluation | The old SWLDA training glob selected 102 RC/Train runs, including 19 IDs in the later frozen manifest. The clean pool contains 83 RC/Train runs. | `results/tables/classifier_protocol_audit.json`; `splits/heldout_manifest.json`. |
| Historical Study D stopping | README records that the stopping-policy sweep selected its threshold on held-out sessions. | Committed `README.md` at `62ce804`, lines covering the historical-result notice. Exact historical sweep artifact lineage is not independently reconstructed here. |
| Held-out-informed outputs | Prior 58.9% headline, old rung 0–4 ladder, Focus Scan, stopping sweep, **the stopping threshold selected by that sweep**, and other old Study D claims are declared superseded in README. The historical stopping threshold was selected using held-out sessions; L7 documents the clean re-selection. | Committed `README.md` at `62ce804`; old artifacts remain in repository history / current results where present. |
| Clean replay | Clean M0: 78.4% fused; old contaminated SWLDA replayed on the identical 57-run manifest with the clean M0 policy: 68.9%. The observed metric is lower, so this audit shows no inflation; because old-model calibration was not refit, it cannot isolate the overlap effect. It is not a reproduction of the historical 58.9% pipeline result. | `results/tables/classifier_manifest_eval.json` (`audit_comparison`); README at `62ce804`. |
| Study D registry exposure | Exact recorded deviation: “The first 300 bytes of data/processed/ground_truth_registry.json were read while checking LM availability. The registry begins with held-out Study D session IDs and target strings, so a small number of held-out labels were exposed to the analyst/model.” The same artifact records `heldout_eeg_epochs_read: false`, `heldout_runs_used_for_tuning_or_model_selection: false`, and `use_in_analysis: false`. This is a documented metadata/label exposure, not a claim of zero leakage. | `results/tables/heldout_protocol_deviation.json`. |
| Study Q | SE003 Test was evaluated once after the `q-inner-final` freeze; a run-once marker and test-once result exist. | `results/tables/study_q_test_once.started`; `results/tables/study_q_test_once.json`; `results/tables/study_q_phase1_gate.json`; `scripts/evaluate_study_q_test_once.py`. |

The repository provides no counterfactual estimate of how much the historic overlap affected any historical metric. Historical inflated-by-amount: **UNVERIFIED**. The defensible comparison is limited to the identical evaluator/policy audit above. The artifact includes a per-condition split of that comparison (§5); hence it is reported rather than marked NOT RUN.

## 5. Experiments and evidence

### Accuracy populations and evaluator mapping

| Accuracy | Runs / subjects | Characters | Alpha | Evaluator / policy | Artifact and interpretation |
|---|---:|---:|---:|---|---|
| D clean-pool OOF M0 EEG-only | 83 / 17 | 498 | 0.0 | Inner-fusion and manifest-style evaluator, τ=.80, min 2, max 10 | `results/tables/phase0_oof_evaluator_parity.json`; both paths report 79.52% and agree on every decision/depth. |
| D clean-pool OOF M0 fused, canonical recomputation | 83 / 17 | 498 | .1 | Shared evaluator, τ=.80, min 2, max 10 | Parity JSON reports 81.93%; `results/tables/classifier_inner_fusion.json` and `results/tables/classifier_nested_calibration.json`. This is internal OOF, not held-out performance. |
| D clean-pool OOF M0 fused, alternate alpha | 83 / 17 | 498 | .85 | Same parity comparison and stopping | Parity JSON reports 84.34%; an alternate-alpha sensitivity point, not the locked deployment alpha. |
| D clean-pool OOF M0 policy re-selection | 83 / 17 | 498 | .1 | Nested run-grouped OOF policy sweep; M0; fixed alpha .1 | `results/tables/ablation_inner_m0_policy_alpha_0p1.json` (SHA-256 `bd153fc503e09598f16bc01254178ebae0cddb1ea7b70033c1e97b0944d322fc`); selected τ=.70, min 2, max 10. Accuracy matched the α=.1 default τ=.80 policy at 81.93%; mean sequences were 5.73 versus 6.45. Training-pool OOF only; not manifest evaluation. |
| D Study D manifest M0 fused / EEG-only | 57 / 17 | 338 total, of which 11 are terminal zero-coverage characters | .1 | One-shot manifest evaluator; lock-defined model and clean M0 stopping | `results/tables/classifier_manifest_eval.json`; 78.40% fused, 72.49% EEG-only. Condition rows follow. |
| D manifest M3a fused / EEG-only | 57 / 17 | Same 338 total, including 11 zero-coverage terminal characters | .1 | Same evaluator and policy | `results/tables/classifier_manifest_eval.json`; 74.26% fused, 60.95% EEG-only. Condition rows follow. |
| D manifest contaminated SWLDA fused / EEG-only | 57 / 17 | Same 338 total, including 11 zero-coverage terminal characters | .1 | Same evaluator, labels, alpha, and clean M0 stopping policy; old calibration not refit | `results/tables/classifier_manifest_eval.json`; 68.93% / 55.62%. Descriptive audit only, excluded from finalist selection and Holm family. Condition rows follow. |
| D clean M0 ablation ladder rung 1 | 57 / 17 | 339 | .85 (locked rung weight) | `scripts/run_clean_m0_ablation_ladder.py`; rung-specific ladder evaluator | `results/tables/ablation_ladder_clean_m0.json`; 75.81% character accuracy. This artifact reports 339 characters and zero zero-coverage characters; the classifier manifest artifact reports 338 total, of which 11 are zero-coverage. The one-character difference is **UNVERIFIED**. Do not merge the populations or treat their accuracies as interchangeable. |
| Q SE003 Test fused / EEG-only | 180 / 36 | 1,296 as-recorded; 396 characters have 10 sequences | .1 | One-shot Q evaluator, naturally variable recorded depth | `results/tables/study_q_test_once.json`; `results/tables/study_q_phase1_gate.json`; 59.41% fused / 55.86% EEG-only. Not comparable to full-depth inner OOF without the recorded-depth caveat. |

For the D manifest result, the artifact combines Dyn and DynBigram into one subgroup. Separate accuracy rows for those conditions are **NOT REPORTED** in that artifact; the run counts are available (Dyn 20, DynBigram 18). Do not infer condition-specific accuracy from the pooled subgroup. This is a condition-stratification reporting limitation.

| Manifest condition | Manifest runs | M0 fused / EEG-only | M3a fused / EEG-only | Contaminated SWLDA fused / EEG-only | Artifact coverage |
|---|---:|---:|---:|---:|---|
| RC/Train | 19 | 86.84% / 79.82% | 81.58% / 69.30% | 70.18% / 51.75% | Reported separately in `classifier_manifest_eval.json`. These 19 run IDs are exactly the 19 RC/Train manifest runs that overlap the old SWLDA training pool. Thus, on the runs old SWLDA had trained on, it scored 70.18% fused versus 86.84% for clean M0, which did not train on them. |
| Dyn | 20 | NOT REPORTED | NOT REPORTED | NOT REPORTED | Pooled with DynBigram in `classifier_manifest_eval.json`. |
| DynBigram | 18 | NOT REPORTED | NOT REPORTED | NOT REPORTED | Pooled with Dyn in `classifier_manifest_eval.json`. |
| Dyn + DynBigram pooled | 38 | 74.11% / 68.75% | 70.54% / 56.70% | 68.30% / 57.59% | Reported as 224 characters, including zero-coverage terminal records. |

### Evidence register

| ID | Question | Method / result | Evidence | Branch / commit | Status |
|---|---|---|---|---|---|
| L1 | Which legacy training runs overlap the D holdout? | Inventory old SWLDA training IDs against manifest; 19 overlaps among 102, leaving 83 clean runs. | `results/tables/classifier_protocol_audit.json` | `optimise`, audit current at `62ce804` | FINAL audit |
| L2 | Does the clean pool support classifier selection? | Nested run-grouped OOF on the clean RC/Train pool; model screen and finalist lock. | `results/tables/classifier_nested_calibration.json`; `splits/finalist_lock.json` | `classifier-final` lineage; final state `62ce804` | FINAL for selection protocol |
| L3 | Did M3a beat M0 on the frozen D manifest? | Paired subject comparison on the one-shot manifest. M3a−M0 fused accuracy: −4.20 pp; 95% subject-bootstrap CI [−10.25, +1.08] pp; Wilcoxon p=.2662, Holm p=.2662. M3a used fewer sequences (2.95 vs 3.82) and had higher variable-time ITR (38.25 vs 32.09 bits/min), but did not improve accuracy; 4 subjects were better, 6 tied, 7 worse. | `results/tables/classifier_manifest_eval.json`; committed README at `62ce804` | Run hash in result: `e16ed223`; final decision at `62ce804` | FINAL; M3a rejected |
| L4 | Did legacy contamination inflate the same-run classifier comparison? | Same 57 manifest runs, labels, alpha, and clean M0 stopping policy: contaminated SWLDA 68.93% vs clean M0 78.40%. On the exact 19 overlapping RC/Train runs, contaminated SWLDA was 70.18% and clean M0 86.84%. The old model's calibration was **not refit** under the clean M0 policy. The audit shows no inflation, but this calibration mismatch means it cannot isolate the overlap effect or quantify historical bias. | `results/tables/classifier_manifest_eval.json:audit_comparison` | Evaluation hash `e16ed223`; final at `62ce804` | FINAL descriptive audit |
| L5 | Do the inner-OOF and manifest-style evaluator paths agree? | They agree on 498 clean OOF characters at α=0, .1, .85: zero decision, stopping-depth, or max-sequence mismatches. At α=.1 both report 81.93%. | `scripts/audit_oof_evaluator_parity.py`; `results/tables/phase0_oof_evaluator_parity.json` | `62ce804` | FINAL parity audit |
| L6 | What is the status of Study Q external validation? | Phase 1 EEG gate passed on training-only CV. SE003 Test evaluated once at as-recorded depth; 1,296 characters; fused 59.41%, EEG 55.86%; no Q stopping, flashes-per-character, or ITR claim. | `results/tables/study_q_phase1_gate.json`; `results/tables/study_q_test_once.json`; `.started` marker | Pre-test freeze `f5010b0`; manifest freeze `449c266`; closeout `62ce804` | FINAL one-shot evaluation |
| L7 | Was the clean M0 stopping policy selected without held-out data? | `policy_selection` uses nested run-grouped OOF with calibration fit on each outer training pool. At fixed base α=.85, M0 selected τ=.80, min sequences 2, max 10 on 498 characters; the 84.34% stopped accuracy has zero decision flips and no accuracy loss versus the default policy on those 498 OOF characters at α=.85. This α=.85-selected policy is the one used in the manifest evaluation, whose finalist lock specifies α=.1. | `results/tables/ablation_inner_m0_policy.json`; `splits/ablation_lock.json`; `splits/finalist_lock.json` | Clean policy selection `62ce804`; manifest lock/evaluation unchanged | FINAL training-only selection |
| L8 | Does the stopping policy selected at α=.85 remain the training-OOF choice at the manifest's α=.1? | Same 498 clean OOF characters, same nested run-grouped OOF, α=.1 re-selection. Selected τ=.70, min 2, max 10 (locked manifest policy: τ=.80, min 2, max 10). Both: 81.93% OOF accuracy. Mean sequences 6.45 → 5.73 (about 11% fewer). Fourteen decisions changed: 6 corrected, 6 regressed, and 2 wrong-to-wrong. Net flips are balanced; there is no accuracy evidence either way. | `results/tables/ablation_inner_m0_policy_alpha_0p1.json`; whole-file SHA-256 `69f5597b549105c28e17ab50de98f8ddf915574d407857c9f32d13c4ed6b8d0c`; selection script SHA-256 `6d56a772d184bf5a1ba426e0b6a96d63e1dd04845ae027958d0d306917c14ab4`; evaluator script SHA-256 `6fb8c91bedf83fe81925a9b857b6486715695c66ad35390eeab06efcea2699b0`; cache SHA-256 `8811ec22ee33a0053e29e02aa9aa862855a1f6514e5d1b22fd5b5332dd2ffa81`; score SHA-256 `d99a52854c4f267d9fee9de6641ac30fb417d9a38d32969d7195c6d51f6daf42` | Generated from `b9e9266`; generated artifact provenance is in JSON | FINAL training-only candidate; NOT applied to or validated on the Study D manifest |

### Evaluator reconciliation

The parity audit **passes** on the identical clean OOF population: 498 characters from 83 runs and 17 subjects. At alpha .1 both evaluator paths report 81.93%; at alpha 0 both report 79.52%; at alpha .85 both report 84.34%. Their scored decisions and stopping depths match (zero mismatches). The often juxtaposed 75.81% and 78.40% values are not a parity disagreement: 78.40% is M0 on the 57-run manifest at alpha .1; 75.81% is clean-M0 ladder rung 1 on the 339-character ladder population at locked rung weight .85. This explanation follows the README, lock, parity JSON, and separate result populations; the one-character 338-vs-339 count discrepancy remains **UNVERIFIED**.

The earlier saved clean OOF M0 alpha-.1 fused accuracy of 87.15% was recomputed as 81.93%. The old input cache had no recorded hash and cannot be verified as byte-identical to the current cache. The old result is quarantined (§6); parity uses the recomputed value.

## 6. Dead arms, superseded results, and quarantined metrics

| Output / choice | Disposition | Evidence / archive |
|---|---|---|
| Historical Study D 58.9% headline and contaminated-pool results | Superseded; overlap plus held-out-tuned stopping prevents treating them as valid held-out estimates. The amount of bias is not established. | Committed `README.md` at `62ce804`; history and prior artifacts. |
| Old Study D rung 0–4 ladder, Focus Scan, stopping sweep, old personalization claims | Superseded; the older personalization table is audit-only. | `results/tables/personalization_ablation_studyd.json`; `results/tables/personalization_ablation_studyd_status.json`; README at `62ce804`. |
| Saved OOF M0 alpha-.1 fused accuracy 87.15% | **QUARANTINED**; recomputed value is 81.93%. Old cache provenance is unverifiable. No paper number uses 87.15%. | `results/tables/phase0_oof_evaluator_parity.json`; `splits/baseline_d_lock.json`; README at `62ce804`. |
| CNN arm | Dead/unavailable as a clean control: metadata audit found Test-labelled training runs; retrain on clean RC/Train before any control claim. | `results/tables/classifier_protocol_audit.json`; branch `CNN`; commits `911c89a`, `38d47fb`. |
| Study D preprocessing/filter historical artifacts | Superseded with the older result path; preprocessing discussion belongs to Component 2 and is not restated here. | `docs/02_preprocessing.md` at `b9e9266` (`preproc-ablation-final`). |

`dead_arms/` contains archived focus-scan and row-focus code/results, including `dead_arms/row_focus_compact_scan.py`, `dead_arms/standard_interleaved_rc.py`, and `dead_arms/results/`. Not every superseded metric has a dedicated archived file; some survive only in README and Git history.

## 7. Final locked configuration

| Lock | SHA-256 | What it freezes | Commit / tag status |
|---|---|---|---|
| `splits/heldout_manifest.json` | `de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b` | D 57-run held-out membership | Manifest freeze in history; `62ce804` consumes it |
| `splits/baseline_d_lock.json` | `bbc663ef45ace7e64d683a9f6824459c2fdd7b70e53d12d9bd031665abda5361` | Baseline D parent locks and OOF audit provenance | `baseline-d-final` is a remote-listed tag; lock records source `494066a` |
| `splits/finalist_lock.json` | `46e7c2008f89c772158e3f11aedb4af799e09b1fbc30a0c628df6c079731509a` | M0 baseline, M3a finalist before manifest evaluation, preprocessing and calibration | Lock identifies pre-evaluation M3a; final choice is M0 after `e16ed22` evaluation |
| `splits/ablation_lock.json` | `840c12db474735757ece08ae70b489795870c09933dfd18819a34a32de895095` | `classifier.id=clean_m0`, rung weights, training-only phrase-bank and 20% salted SHA-256 OOF weight-fit split | Created at `fe044b6`; ladder result at `e7b133c`; consumed by `62ce804` |
| `splits/study_q_manifest.json` | `39ed6801b2545f37f5f0b1019c01bec0e8fd6b7a6b4418c7014c00b9c689c973` | Q SE003 Test membership | Freeze commit `449c266`; `q-manifest-frozen` local-only |
| `splits/study_q_excluded_training_runs.json` | `f267f7751339dff9c1b214521c562ca98258846bfc2420a95d26f0b10817f3d1` | Q same-session Train exclusions | `449c266`; `q-manifest-frozen` local-only |
| `splits/studyq_classifier_lock.json` | `f43483c3b5a9605bdef927c0ee6ed5010feabe1dcdb02ca7b96b6ebe40e19f44` | Q locked classifier recipe and one-shot SE003 protocol | Pre-test state at `f5010b0`; `q-inner-final` local-only |
| `splits/rag_lock.json` | `169b758421233410b2ec0ff6c5cf348a29cbf22f276c624a94e5f42394937fdd` | Prospective RAG corpus/split and gates; G2 no-go for live use | In committed state `62ce804` |
| `splits/scheduler_lock.json` | `eac9be209c4447c3854a38ae07a21e8e93a667b26bbe8287df460500f819b61c` | Scheduler simulation selection boundaries; current G3 no-go | In committed state `62ce804` |

Paper-final classifier: **clean M0**. The pre-evaluation finalist lock named M3a, but one-shot paired manifest evidence found no significant accuracy gain for M3a; it used fewer sequences and had higher variable-time ITR, while the accuracy point estimate was lower. The committed README decision retains M0. Per the latest user-verified origin listing, remote-listed tags are `ablation-final`, `baseline-d-final`, `classifier-final`, and `preproc-ablation-final`. `phase3-final` at `62ce804`, `q-inner-final` at `f5010b0`, and `q-manifest-frozen` at `449c266` are local-only tags.

## 8. Validity rules for later components

- Fit classifiers and calibrators on the clean training pool only; use run-grouped OOF for model/policy selection.
- Study D training remains RC/Train-only. Report its manifest evaluation by RC/Train, Dyn, and DynBigram whenever artifact support exists. Current classifier-manifest result pools Dyn and DynBigram; separate condition accuracies are **NOT REPORTED** there.
- Phrase-bank construction excludes deterministic internal-validation sessions and all manifest sessions. Fusion-weight fitting uses the locked training-run OOF validation split; it must not use manifest labels.
- Held-out EEG/labels may be accessed only by the declared one-shot evaluator after verifying manifest and lock hashes. Record access in evaluator outputs and one-shot markers. Do not tune after observing a held-out result.
- Study Q Test supports the stated as-recorded-depth validation only; the lock bars scheduler tuning and claims of Q stopping, flashes per character, and ITR.
- The first 300 bytes of the Study D registry were read, as quoted in §4. The artifact records no held-out EEG epoch reads and no held-out tuning/model selection. Compliance for other historical analyses is **UNVERIFIED**; registry bytes included held-out IDs and target strings, so label text exposure is documented.
- The Study D same-run audit is descriptive and does not quantify counterfactual historical bias.

## 9. Evaluator reconciliation

See §5. The two OOF evaluator paths match on all 498 compared characters at alpha 0, .1, and .85, with zero decision and stopping-depth mismatches. Differences between the 75.8% ladder reference and 78.4% manifest accuracy correspond to different populations and configurations, not a remaining OOF evaluator disagreement. The manifest artifact's pooled Dyn/DynBigram subgroup and the one-character difference between the 338-character manifest total and 339-character ladder total remain explicit limitations.

## 10. Limitations and open items

- The classifier manifest has 338 total characters, of which 11 are zero-coverage; the ladder reports 339 characters. The one-character difference is not reconciled by the current artifacts (**UNVERIFIED**).
- At α=.1, clean OOF re-selection chose τ=.70, min 2, max 10, while the one-shot manifest evaluation used locked τ=.80, min 2, max 10. Both policies scored 81.93% on the 498 clean OOF characters at α=.1, but τ=.70 used fewer mean sequences (5.73 vs 6.45). The α=.1-selected policy was not evaluated on the manifest; the locked manifest result remains the pre-registered τ=.80 result. This clean OOF finding supports a candidate policy but does not revise the completed one-shot evaluation.
- The classifier manifest artifact reports Dyn/DynBigram jointly; per-condition accuracies for those two conditions are **NOT REPORTED**.
- No counterfactual establishes the magnitude or direction of historical overlap bias. Only the identical-evaluator 68.9% vs 78.4% audit is available.
- Historical compliance outside the documented registry exposure is **UNVERIFIED**.
- The Study D manifest comparison JSON records run commit `e16ed223` and lock/manifest hashes but does not include a training-cache SHA; exact byte provenance of its input cache is **UNVERIFIED**.
- Remote tag state follows the latest user-verified origin listing; a live remote check was unavailable during this turn.

## 11. Reproduction

Commands below use only parser-confirmed arguments. Dataset and cache inputs are external and are not committed; do not rerun one-shot held-out evaluators as routine reproduction.

| Purpose | Command / script | Inputs and outputs |
|---|---|---|
| Recompute D protocol overlap and counts | `python3 scripts/audit_classifier_protocol.py` — script has no argparse CLI | Reads processed epoch files and `splits/heldout_manifest.json`; writes `results/tables/classifier_protocol_audit.json`. Requires external processed data. |
| Verify D cache selection behavior | `python3 scripts/build_classifier_cache.py --epochs data/processed/StudyD --manifest splits/heldout_manifest.json --output data/cache/classifier_epochs.h5 --source epochs` | Options verified in argparse; output is a training-only cache. Default preprocessing options are parser-defined. Requires external epochs. |
| Audit Study Q path-derived split | `python3 scripts/audit_study_q.py` | With frozen files, validates inventory; `--refreeze` exists but is inappropriate after evaluation. Requires external Study Q EDF inventory. |
| Audit evaluator parity | `python3 scripts/audit_oof_evaluator_parity.py --scores results/tables/classifier_nested_calibration_scores.npz --cache data/cache/classifier_epochs.h5 --output results/tables/phase0_oof_evaluator_parity.json` | Parser-confirmed `--scores`, `--cache`, `--output`; requires the exact cache and score inputs, whose hashes are in the parity artifact. |
| Re-select D M0 stopping policy at manifest alpha | `MPLCONFIGDIR=/private/tmp/policy-alpha1-mpl .venv/bin/python scripts/reselect_m0_policy_alpha_0p1.py --output results/tables/ablation_inner_m0_policy_alpha_0p1.json` | Wrapper fixes α=.1, M0, and the 24-point stopping grid; only `--output` is configurable. Records wrapper/evaluator/input hashes in the result. The sweep selects and scores on the same OOF population; no interval was calculated. |
| Inspect one-shot evaluator CLI | `scripts/evaluate_classifier_manifest.py` | Confirmed parser flags include `--allow-heldout-eval`, `--dry-run-training`, `--lock`, `--manifest`, `--cache`, `--config`, `--raw-dir`, `--old-model`, `--output`. It was already run; do not rerun the one-shot replay. |
| Inspect Study Q test-once CLI | `scripts/evaluate_study_q_test_once.py` | Confirmed flags `--allow-heldout-eval`, `--dry-run-train`; it is guarded by an exclusive marker and was already run. Do not rerun. |
| Validate frozen digests without accessing data | `shasum -a 256 splits/heldout_manifest.json splits/study_q_manifest.json splits/baseline_d_lock.json splits/finalist_lock.json splits/ablation_lock.json splits/studyq_classifier_lock.json` | Compare output to the SHA-256 values in §§2 and 7. |

## 12. Handoff to Component 4 — classifier

1. Study D clean training pool: 83 RC/Train runs, 17 subjects; 84,660 epochs (9,960 target, 74,700 non-target).
2. Study D held-out manifest: 57 runs; 19 RC/Train, 20 Dyn, 18 DynBigram; SHA-256 `de2fd72215e6f2766cc66c629ebd08fd9a1c55b6dab40c63a2e7890a0a8bac7b`.
3. Nineteen old SWLDA training runs overlap the manifest; old pool 102, clean pool 83.
4. Paper-final Study D classifier is clean M0; manifest: 78.40% fused, 72.49% EEG-only on 338 total characters, of which 11 are zero-coverage.
5. M3a manifest result: 74.26% fused; paired delta −4.20 pp (95% CI −10.25 to +1.08), Holm p=.2662; not selected.
6. Contaminated SWLDA same-run audit: 68.93% fused vs M0 78.40% under the identical evaluator/policy; historical 58.9% is unreproduced and superseded.
7. OOF parity passes for 498 clean characters at alpha 0, .1, .85; zero decision/stopping mismatches.
8. Study Q clean train pool: 360 runs; SE003 Test manifest: 180 runs, 36 subjects, 1,296 as-recorded characters; test-once evaluation complete.
9. Registry exposure: first 300 bytes read; no held-out EEG epochs read and no tuning/model selection, per deviation artifact; other historical compliance UNVERIFIED.
10. Component 2 handoff (`docs/02_preprocessing.md`, `b9e9266`, `preproc-ablation-final`):
    1. Parser output: MNE FIF epochs `X=(epochs, EEG channels, samples)`, binary event label `0=non-target`, `1=target`.
    2. Per-epoch metadata: `stimulus_code`, `stimulus_type`; adaptive adds 0-based `char_index` and 1-based `sequence_in_char`; group flash adds row-major `lit_mask`.
    3. HDF5 is a separate classifier-cache interface; cache schema is deferred to the classifier component.
    4. Batch parser/FIF defaults: 0.1–30 Hz, no notch, −0.1…0.8 s, baseline on, no rejection, decim 1.
    5. Separate clean-reference cache `classifier_epochs.h5`: 83 RC/Train runs, 84,660 epochs, 0.1–30 Hz, baseline on; SHA is in `splits/baseline_d_lock.json`.
    6. Paper-final Study D classifier M0 uses effective 1–12 Hz, baseline off, classifier window 0…0.8 s, decim 2; direct raw-EDF parse is pinned by `splits/finalist_lock.json` and result JSON. M3a was the pre-evaluation finalist and was not adopted.
    7. Study Q cache: 1–12 Hz, −0.1…0.8 s, baseline off, decim 2, 12-flash group sequence.
    8. Study D training labels: 9×8 grid; 9,960 targets vs 74,700 non-targets (1:7.5).
    9. Baseline-off M1 comparison: 1–12 vs 0.1–30 Hz AUC 0.7599 vs 0.7350; character accuracy 67.07% vs 62.05% (498 characters).
    10. P7/P8 matched M1 CIs span zero: P7 filtered-minus-unfiltered AUC −0.40 to +0.93 pp and character accuracy −7.23 to +1.20 pp; P8 baseline-on-minus-off AUC −0.35 to +0.98 pp and character accuracy −6.63 to +2.81 pp. P9 1–12 vs 0.1–30 Hz CIs are AUC +1.94 to +3.06 pp and accuracy +0.80 to +9.24 pp; all are M1 evidence, not paper-final M0.
