# 08 — Evaluation and Statistics Protocol

## 1. Purpose, scope, and evidence tiers

This protocol defines how the paper reports an evaluated claim. It does not make development, held-out, simulation, and software-check evidence interchangeable.

| Tier | Permitted claim | Evidence boundary |
|---|---|---|
| Nested OOF development | A model/policy behaved as recorded on its training-pool nested out-of-fold evaluation. | Selection/development only; not a held-out estimate. |
| One-shot manifest | A locked method obtained the recorded result on the frozen Study D manifest. | One-shot; do not rerun for selection. |
| Study Q training-only | A method behaved as recorded on Study Q session-grouped nested OOF. | Development only. |
| Study Q one-shot test | A locked method obtained the recorded as-recorded Test result. | One-shot; no stopping, flashes/character, or ITR claim unless the artifact defines it. |
| Simulation | The simulator produced the recorded synthetic result. | “Simulated; not measured EEG.” |
| Software check | Code/lock/leakage behavior passed a local test. | Not an experimental outcome or paper efficacy result. |

A paper **claim** is a numerical or comparative statement about a method, population, or outcome. Every claim must identify its tier, population, analysis unit, artifact, and estimator. Do not compare numbers across tiers, alpha settings, evaluators, or populations as if they were paired estimates.

## 2. Metric definitions

| Metric | Definition and code location | Unit | Improvement | Notes |
|---|---|---|---|---|
| Character accuracy | `correct_characters / max(1, total_characters)` (`src/evaluation/metrics.py::SpellerMetrics.accuracy`); evaluators aggregate decoded usable character records. | Character; inferential aggregate is subject. | Higher | Manifest evaluator reports zero-coverage terminal characters separately and excludes them from scored records. |
| Flashes/character | `total_flashes_used / max(1, total_characters)` (`SpellerMetrics.average_flashes_per_char`). | Flash count per character. | Lower | Use recorded flash count; terminal zero-coverage handling is evaluator-specific. |
| Time/character | `total_time_seconds / max(1, total_characters)` (`SpellerMetrics.time_per_character`). | Seconds/character. | Lower | Time model is Component 5 / `src/evaluation/sequence_scoring.py` evaluator path. |
| Variable-time ITR | Wolpaw bits: `log2(N)+p log2(p)+(1-p)log2((1-p)/(N-1))`, boundary terms omitted, floored at zero, times `60/T`; `N=72` for D. | Bits/minute. | Higher | `src/evaluation/sequence_scoring.py::variable_time_itr`; observed mean seconds/character, not a fixed-trial duration. |
| WPM | `(60 / time_per_character) / 5`. | Words/minute. | Higher | This is the paper definition: `SpellerMetrics.wpm`. |
| WPM approximation (ITR/5) | `itr_bits_per_minute / 5`. | ITR-derived approximation. | Higher | Used as `wpm_approx` by manifest/ladder evaluators. It is **not** the WPM definition above. |
| Epoch AUC | ROC AUC on per-flash target scores. | Epoch/flash. | Higher | Preprocessing bootstrap recomputes paired pooled AUC from run-pair win counts. |
| Row/column sequence NLL | Mean sequence row/column negative log likelihood from evaluator records. | Sequence; inferential aggregate is subject. | Lower | Secondary in the D manifest finalist analysis. |
| Calibration NLL / ECE | Binary `log_loss(y, sigmoid(clipped logits))`; ECE is the 15-bin probability-weighted absolute calibration gap. | Flash/epoch. | Lower | `src/evaluation/calibration.py::calibration_metrics`; logits clipped to `[-60,60]`. |

Stopping is implemented by `SequentialDecoder`: accumulate log grid posteriors, stop after at least `min_sequences` when maximum posterior reaches `tau`, otherwise stop at `max_sequences` (`src/evaluation/sequence_scoring.py`). Component 5 owns policy selection and the time model.

## 3. Populations, strata, and exclusions

| Population | Scope | Reportable strata / exclusions | Evidence |
|---|---|---|---|
| Study D clean nested OOF | 83 RC/Train runs, 17 subjects, 498 characters. | Training-pool development only. | `results/tables/classifier_nested_calibration.json`; `docs/04_classifier.md`. |
| Study D manifest ladder | `manifest_57_runs`, 17 subjects, 339 characters per rung. | Overall; `RC/Train` (114); `Dyn/DynBigram` (225); zero coverage 0. | `splits/ablation_lock.json`; `results/tables/ablation_ladder_clean_m0.json`. |
| Study D finalist manifest | 57 runs, 17 subjects, 338 scored characters. | Overall and evaluator strata; 11 zero-coverage terminal characters reported separately. | `results/tables/classifier_manifest_eval.json`. |
| Study Q nested OOF | 36 subjects, 2,592 characters. | Session-grouped training-only OOF. | `results/tables/study_q_nested_fusion*.json`. |
| Study Q SE003 Test | 180 runs, 36 subjects, 1,296 as-recorded characters. | Overall and condition strata; stopping/ITR/flashes claims prohibited by artifact. | `results/tables/study_q_test_once.json`. |
| Preprocessing probe | 83 runs, 84,660 epochs, 498 characters. | M1 paired OOF score files only. | `results/tables/preproc_ablation_filter_bootstrap*.json`. |

Zero-coverage terminal characters must be displayed as a count alongside the scored-character denominator. The 338-versus-339 discrepancy is an open item; its cause is **UNVERIFIED**. Do not pool, pair, or normalize the two denominators.

Numbers from distinct populations, alpha values, stopping evaluators, or evidence tiers never share a statistical comparison. In particular, D alpha `.85` clean-OOF selection, D alpha `.1` candidate replay, and Q Test recorded-depth evaluation are separate analyses.

## 4. Units of analysis and aggregation

| Quantity | Descriptive unit | Inferential/resampling unit | Code/artifact convention |
|---|---|---|---|
| Decoding accuracy, flash count, time, ITR | Character / sequence / flash as defined above | Subject for locked D/Q paired method comparisons | Subject rates are paired before Wilcoxon and subject bootstrap. |
| Preprocessing AUC and accuracy | Epoch AUC; character accuracy | Run cluster | Paired `run_id` resamples preserve candidate/reference pairing. |
| Classical classifier screen | Character rates within subject | Subject, with within-subject character resampling | Hierarchical bootstrap. |
| Calibration | Flash/epoch | **UNVERIFIED** for paper-level inference | Code defines metrics; do not imply a CI/test absent from artifact. |

Subject is the inferential unit for manifest comparisons because the locked evaluators form one aggregate value per shared subject before paired testing. Run is the preprocessing bootstrap cluster because its code samples `run_id` with replacement and recomputes each paired condition’s aggregate. These are code/lock conventions, not claims about a universal sampling model.

## 5. Inferential methods

`src/evaluation/ablations.py::_paired_wilcoxon` defines differences as **candidate/comparison minus baseline**, uses a two-sided SciPy Wilcoxon test with `zero_method="wilcox"` and `method="auto"`, requires at least two pairs, and assigns statistic 0 and p=1 for all-zero differences. Positive accuracy/ITR deltas favor the candidate; negative flashes/character deltas favor the candidate. Individual scripts may use the same numerical convention without importing this helper; report their actual script.

| Analysis | Unit resampled | Replicates | Seed | CI | Test | Multiplicity family | Artifact |
|---|---:|---:|---:|---|---|---|---|
| D clean-M0 ladder | Subject | 10,000 | 0 | Percentile 95% mean-difference CI | Paired Wilcoxon | Five planned rung comparisons | `ablation_ladder_clean_m0.json` |
| D manifest finalist | Subject | 20,000 | 42 | Percentile 95% mean-difference CI | Two-sided paired Wilcoxon, zero_method Wilcox | Up to two primary finalist-vs-M0 fused-accuracy comparisons; NLL separate | `classifier_manifest_eval.json` |
| Classical screen | Subject + within-subject character outcomes | 20,000 | 407 | Percentile 95% CI | Two-sided paired Wilcoxon; rank-biserial reported | All four model/config comparisons | `classifier_classical_stats.json` |
| Preprocessing P7–P9 | Paired `run_id` | 10,000 | 0 | Percentile 95% CI | No null test | None | `preproc_ablation_filter_bootstrap*.json` |
| D inner fusion | Subject | **UNVERIFIED** | 42 | Subject-resampling 95% CI | Paired Wilcoxon | “Holm across all comparisons” | `classifier_inner_fusion_stats.json` |
| Q nested fusion | Subject | 20,000 | 104729 | Percentile 95% mean-difference CI | Two-sided Wilcoxon, `method="auto"` | None recorded | `study_q_nested_fusion_stats.json` |
| Q one-shot Test | Subject | 10,000 | 1907 | Percentile 95% mean-difference CI | Two-sided paired Wilcoxon, zero_method Wilcox | Holm over evaluator condition strata | `study_q_test_once.json` |
| D/Q summary | Subject, separately within study | 30,000 | 271828 | Percentile 95% CI of equal-study pooled mean | Two-sided Wilcoxon | Holm over two study tests | `dq_inner_fusion_effect.json` |

The D/Q summary is a **non-standard pooled summary**: code resamples each study separately, averages the two study means with equal study weight, and tests D and Q separately. It does not create a common participant population.

## 6. Locks, guards, and enforcement

| Control | What it freezes or guards | Enforcement |
|---|---|---|
| `splits/ablation_lock.json` | 57-run ladder, rungs, metrics, strata, five-test family, seed 0 / 10,000 bootstrap, one-shot guard. | `tests/test_ablation_lock.py`; ladder script requires `--allow-heldout-eval`. |
| `splits/finalist_lock.json` | D manifest evaluator configuration and parent manifest relation. | `tests/test_finalist_lock.py`; manifest script requires `--allow-heldout-eval`. |
| `splits/baseline_d_lock.json` | D baseline/provenance lineage. | Lock/hash inspection; exact statistical role is artifact-specific. |
| `splits/rag_lock.json` | RAG development/participant protocol. | RAG protocol tests; not a substitute for a final held-out effect estimate. |
| `splits/scheduler_lock.json` | Future simulation modes, lambda grid, target-blindness and no-go conditions. | `tests/test_scheduler_leakage.py`; G3 is no-go. |
| `splits/studyq_classifier_lock.json` | Study Q training exclusions and SE003 one-shot protocol. | `tests/test_classifier_protocol.py`, `tests/test_label_vault.py`, `tests/test_study_q_audit.py`. |
| Label vault and manifests | Held-out/Test labels are inaccessible to training paths by default. | `tests/test_label_vault.py`, `tests/test_classifier_protocol.py`. |

`tests/test_clean_m0_ablation_ladder.py`, `test_sequence_scoring.py`, `test_row_column_posterior.py`, and `test_group_flash_scoring.py` validate scoring behavior. `test_participant_text_protocol.py` validates chronological/text-bank separation. Test execution is local software evidence unless a committed result artifact records it.

Known deviations/open controls: legacy dynamic stopping used held-out data and is superseded; Q Test has a run-once marker; one-shot evaluators are not routine rerun commands.

## 7. Power and interpretation limits

| Analysis | Subjects | Non-tied pairs / attainable two-sided Wilcoxon p | Interpretation limit |
|---|---:|---|---|
| D ladder 1v0 | 17 | 13 non-tied; minimum exact sign-pattern probability `2/2^13 = 0.000244`; Wilcoxon minimum depends on rank distribution: **UNVERIFIED** | Four ties reduce effective information. |
| D ladder 2v1 / 4v1 | 17 | 10 non-tied; sign-pattern lower bound `2/2^10 = 0.001953`; exact Wilcoxon minimum **UNVERIFIED** | Seven ties. |
| D ladder 3v1 | 17 | 7 non-tied; sign-pattern lower bound `2/2^7 = 0.015625`; exact Wilcoxon minimum **UNVERIFIED** | Ten ties. |
| D ladder 4v2 | 17 | 0 non-tied; all-zero convention p=1 | No directional evidence. |
| D manifest finalist | 17 | 11 non-tied (4 better, 6 tied, 7 worse); sign-pattern lower bound `2/2^11 = 0.000977`; exact Wilcoxon minimum **UNVERIFIED** | Six ties. |
| Study Q nested fusion | 36 | 36 improved, 0 tied, 0 worse; observed Wilcoxon p recorded | Training-only result. |

The sign-pattern calculations are lower-bound reference quantities from the recorded number of non-tied subject deltas; they are not substituted Wilcoxon p-values. No equivalence or non-inferiority test is implemented. “Not significant” means the stated test/family did not reject; it does not establish no effect. Multiplicity is corrected only within each declared analysis family; it is not globally corrected across studies, tiers, artifacts, or post hoc summaries unless an artifact states otherwise.

## 8. Convention inventory and reconciliation

Every row gives direction as candidate minus reference; missing metadata is explicitly marked.

| Result / evidence tier | Unit; n | Test; family | CI; replicates; seed | Direction | Reconciliation |
|---|---|---|---|---|---|
| Preprocessing P7–P9 / nested OOF | Run; 83 runs, 498 characters | None; none | Run-cluster percentile 95%; 10,000; 0 | Candidate − reference | Acceptable-by-design: run resampling. |
| Classical classifier / OOF screen | Subject; 17 | Paired Wilcoxon; Holm over 4 | Hierarchical subject/character percentile 95%; 20,000; 407 | Candidate model − M0 | Exploratory/superseded 249-character grouping must not be used as final evidence. |
| D inner fusion / nested OOF | Subject; 17 | Paired Wilcoxon; Holm over all comparisons | Subject bootstrap 95%; **UNVERIFIED** draws; 42 | Named model − comparator | Open metadata gap: replicate count absent. |
| D ladder / one-shot manifest | Subject; 17 | Paired Wilcoxon; five rung comparisons | Subject percentile 95%; 10,000; 0 | Candidate rung − baseline rung | Final locked ladder. |
| D finalist / one-shot manifest | Subject; 17 | Paired Wilcoxon; max 2 fused-accuracy finalists | Subject percentile 95%; 20,000; 42 | M3a − M0 | Final decision retained M0. |
| Q nested fusion / training-only | Subject; 36 | Paired Wilcoxon; none recorded | Subject percentile 95%; 20,000; 104729 | Fusion − EEG | Do not compare with Q Test as paired tier evidence. |
| Q SE003 Test / one-shot | Subject; 36; 1,296 characters | Paired Wilcoxon; Holm over conditions | Subject percentile 95%; 10,000; 1907 | Fusion − EEG | As-recorded depth; artifact prohibits stopping/ITR/flashes claims. |
| D/Q fusion summary / cross-study summary | Subject within each study; D=17, Q=36 | Separate Wilcoxon; Holm over 2 studies | Equal-study stratified percentile 95%; 30,000; 271828 | Fusion − EEG | Non-standard pooled summary; use only with explicit label. |
| RAG benchmarks/sweeps / development | **UNVERIFIED** | **UNVERIFIED** | Artifact-specific; **UNVERIFIED** | Artifact-specific | Do not promote dry run/sweep to held-out claim. |
| Stopping replay / development | Character; D=498 / Q=2,592 where stated | Policy replay, not universal hypothesis family | **UNVERIFIED** unless artifact records it | Policy − comparator | Alpha `.85` and `.1` are distinct. |
| EEG simulator / simulation | Subject/character as artifact records; D=17, Q=36 | Curve/known-result gates; no multiplicity family recorded | 20 simulation repeats; 20261008 | Simulated − observed/reference | G3 no-go; not measured EEG. |
| `personalization_ablation_studyd.json` / legacy | **UNVERIFIED** | Legacy tracker analysis; **UNVERIFIED** | **UNVERIFIED** | Tracker-specific | **LEGACY/SUPERSEDED**; own status `superseded_audit_only`. |

Inconsistencies: (1) subject/run/hierarchical resampling and seeds differ by analysis—acceptable-by-design if fully named; (2) 338 versus 339 remains open; (3) alpha `.85` versus `.1` are separate policy analyses; (4) `ITR/5` must be labeled “WPM approximation (ITR/5)”, distinct from time-derived WPM; (5) the rung-4 `PERSONALIZATION_RUNGS` label is stale relative to frozen `personalized_gated_rag`—open code-label issue.

## 9. Paper reporting standards

| Claim type | Required accompanying fields | Permitted wording |
|---|---|---|
| Nested OOF development | Tier, population, character/run/subject n, unit, method, CI/test/family, seed/draws, artifact. | “In nested OOF development…” |
| One-shot manifest | Lock, held-out population, zero coverage, subject n, test/family, CI/draws/seed, artifact. | “On the frozen one-shot manifest…” |
| Study Q one-shot | Test status, as-recorded denominator, subject n, condition, test/family, CI/draws/seed. | “On the one-shot SE003 Test…” |
| Simulation | “Simulated,” input provenance, repeats/seed, gate and result. | “The simulator…” never “participants/EEG showed.” |
| Null result | Full point estimate, CI, p and correction family, n/unit. | “Did not reject under the prespecified Holm family”; never “no effect.” |

Checklist for every table/figure:

- State evidence tier, population, strata, exclusions, and denominator.
- State the analysis and inferential units.
- State direction as candidate minus reference.
- State CI type, replicate count, seed, test, and multiplicity family; write `UNVERIFIED` if absent.
- Label zero-coverage counts and WPM approximation separately from WPM.
- Do not put different population/evaluator/alpha/tier values in a common comparison or difference plot.
- Link the committed artifact and lock/commit where applicable.

**RECOMMENDATION:** reserve the equal-study D/Q pooled summary for a supplement and foreground the two study-specific effects. **RECOMMENDATION:** add replicate count to future D inner-fusion artifacts.

## 10. Experiments and evidence register

| ID | Question | Method | Artifact / commit | Status |
|---|---|---|---|---|
| E1a | Does 1–12 Hz baseline-off change M1 OOF scores? | Paired run bootstrap P9 | `preproc_ablation_filter_bootstrap.json`; `preproc-ablation-final` / `b9e9266` | FINAL probe |
| E1b | Does bandpass change M1 OOF scores? | Paired run bootstrap P7 | `preproc_ablation_filter_bootstrap_p7.json`; `preproc-ablation-final` / `b9e9266` | FINAL probe |
| E1c | Does baseline correction change M1 OOF scores? | Paired run bootstrap P8 | `preproc_ablation_filter_bootstrap_p8.json`; `preproc-ablation-final` / `b9e9266` | FINAL probe |
| E2 | Do classical alternatives differ from M0? | OOF paired subject analysis | `classifier_classical_stats.json` | SUPERSEDED / exploratory |
| E3 | Which D fusion/classifier screens merit final testing? | Nested OOF | `classifier_inner_fusion_stats.json` | Development only |
| E4 | Does locked clean-M0 ladder improve D manifest accuracy? | One-shot ladder | `ablation_ladder_clean_m0.json`; `ablation-final` / `494066a` | FINAL |
| E5 | Does M3a improve frozen D manifest accuracy? | One-shot finalist test | `classifier_manifest_eval.json`; `baseline-d-final` / `88d19ec` | FINAL; M0 retained |
| E6 | Does Q fusion help in nested OOF? | Session-grouped nested OOF | `study_q_nested_fusion_stats.json`; `q-inner-final` / `f5010b0` | FINAL development evidence |
| E7 | Does Q fusion help on SE003 Test? | One-shot as-recorded test | `study_q_test_once.json`; `q-manifest-frozen` / `449c266` | FINAL one-shot |
| E8 | What is the cross-study fusion summary? | Equal-study resampling / separate tests | `dq_inner_fusion_effect.json` | FINAL descriptive non-standard summary |
| E9 | Do personalized RAG rungs help in legacy analysis? | Legacy tracker | `personalization_ablation_studyd.json` | LEGACY / SUPERSEDED |
| E10a | Does D alpha `.85` clean OOF support the policy? | Training-only replay | `ablation_inner_m0_policy.json` | FINAL selection evidence |
| E10b | Does D alpha `.1` alter the clean OOF policy? | Training-only replay | `ablation_inner_m0_policy_alpha_0p1.json` | CANDIDATE |
| E10c | Does Q nested OOF select a policy? | Training-only replay | `study_q_nested_fusion.json` | CANDIDATE |
| E10d | Does simulation validate an adaptive scheduler? | D/Q simulator LOSO | `eeg_simulator_loso_validation.json`; `62ce804` | FINAL no-go evidence |
| E11 | Do four schedulers save flashes at matched accuracy? | Locked future simulation | `scheduler_lock.json` | NOT RUN |
| E12a | Does the RAG development benchmark establish final effectiveness? | Development benchmark | `rag_development_benchmark.json` | Development only |
| E12b | Does the RAG parameter sweep establish final effectiveness? | Development sweep | `rag_sweep.json` | Development only |
| E12c | Does self-generated RAG dry run establish final effectiveness? | Dry run | `rag_self_generated_dry_run.json` | NOT RUN for outcome claim |
| E13 | What did legacy held-out dynamic stopping report? | Historical replay | `dynamic_stopping_threshold_sweep.json` | SUPERSEDED |

## 11. Limitations and open items

- 338 versus 339 Study D character populations: **UNVERIFIED** cause; no hypothesis supplied.
- The D inner-fusion bootstrap replicate count is **UNVERIFIED**.
- `PERSONALIZATION_RUNGS` rung-4 text is stale; frozen lock controls the definition.
- WPM and WPM approximation are distinct code paths.
- No equivalence/non-inferiority tests are present.
- Simulation omits temporal dependence, adaptation, fatigue, and nonstationary artifacts; D known-result validation failed.
- Remote tag verification in this session is **UNVERIFIED** because `github.com` DNS resolution failed.

## 12. Reproduction and provenance

| Script | Verified interface | Boundary |
|---|---|---|
| `scripts/bootstrap_preprocessing_ablation.py` | `python scripts/bootstrap_preprocessing_ablation.py --reference PATH --candidate PATH --output PATH --reference-label LABEL --candidate-label LABEL [--n-bootstrap N] [--seed N]` | Paired OOF score files only. |
| `scripts/summarize_study_q_nested_fusion.py` | `python scripts/summarize_study_q_nested_fusion.py [--input PATH] [--output PATH]` | Rewrites summary output. |
| `scripts/summarize_dq_fusion_effect.py` | `python scripts/summarize_dq_fusion_effect.py [--d PATH] [--q PATH] [--output PATH]` | Non-standard cross-study summary. |
| `scripts/analyze_classical_stats.py` | No CLI; reads fixed table inputs and writes `classifier_classical_stats.json`. | Existing artifact is the evidence record. |
| `scripts/evaluate_classifier_manifest.py` | `python scripts/evaluate_classifier_manifest.py --allow-heldout-eval [--lock PATH --manifest PATH --cache PATH --config PATH --raw-dir PATH --old-model PATH --output PATH]` | One-shot evaluator; not routine reproduction. |
| `scripts/run_clean_m0_ablation_ladder.py` | `python scripts/run_clean_m0_ablation_ladder.py --allow-heldout-eval [--lock PATH --manifest PATH --cache PATH --raw-dir PATH --output PATH]` | One-shot evaluator; not routine reproduction. |
| `scripts/evaluate_study_q_test_once.py` | `python scripts/evaluate_study_q_test_once.py --allow-heldout-eval [--dry-run-train]` | One-shot evaluator; not routine reproduction. |
| `scripts/validate_eeg_simulator.py` | No CLI; `python scripts/validate_eeg_simulator.py`. | Depends on external cache inputs; do not use for model selection after observed result. |

Local tag objects: `ablation-final`=`494066a`, `baseline-d-final`=`88d19ec`, `classifier-final`=`dbf5293`, and `preproc-ablation-final`=`b9e9266`. `docs/07_stopping_scheduling.md` records a 2026-11-18 `git ls-remote --tags origin` listing for those four tags. It records `phase3-final`, `q-inner-final`, and `q-manifest-frozen` as local-only; cite `62ce804`, `f5010b0`, and `449c266`. This session’s live remote check is **UNVERIFIED** (DNS failure).

## 13. Handoff block

```text
1. Treat nested OOF, one-shot manifest, Study Q one-shot, simulation, and software checks as separate evidence tiers.
2. Paper WPM is time-derived: (60 / seconds-per-character) / 5.
3. Label evaluator ITR/5 as “WPM approximation (ITR/5)”, never plain WPM.
4. Locked D ladder: 57 runs, 17 subjects, 339 characters/rung; five-comparison Holm family.
5. D finalist manifest: 338 scored characters plus 11 zero-coverage terminals; M3a−M0 is −4.20 pp, Holm p=.2662.
6. The 338-versus-339 cause is UNVERIFIED; do not reconcile or compare populations.
7. Study Q nested OOF is training-only; SE003 Test is a distinct one-shot as-recorded analysis.
8. personalization_ablation_studyd.json is LEGACY/SUPERSEDED; its rung-4 label is stale versus the frozen lock.
9. D/Q equal-study pooling is a non-standard descriptive summary, not a common participant population.
10. Remote tag check this session is UNVERIFIED; use docs/07’s dated listing and local commit objects.
```
