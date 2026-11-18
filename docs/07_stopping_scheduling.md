# Component 7 — Stopping and flashing scheduling

## 1. Purpose, inputs, and conclusion

**Question.** Can the system use fewer flashes per character without losing character accuracy?

**Inputs.** Clean Study-D and Study-Q OOF classifier-score artifacts; the Component 5 decoder/stopping and time model; the language/RAG priors from Components 5–6; archived pre-audit row-focus replays; and `splits/scheduler_lock.json`.

**Outputs.** A paper/live stopping-policy lock, archived dead-arm accounting, a target-blind scheduler prototype, a preregistered scheduler-simulation protocol, and the G3 decision.

**Answer at this state.** Dynamic stopping is locked for the paper, but its policy-selection evidence is clean OOF rather than a new flashing experiment. The alpha=.1 candidate reduces mean sequences without a pooled OOF accuracy loss, but it was not manifest-evaluated. The language-guided scheduler has no outcome simulation, matched-accuracy flash-saving, ITR, false-lock-in, or unlikely-target recovery result. G3 is **no-go**; live flashing remains uniform. [Component 5](05_language_model_prior.md) defines the stopping rule, decoder, and variable-time ITR and is not re-derived here.

**State and provenance.** The checked worktree was clean. Remote tags checked with `git ls-remote --tags origin` on 2026-11-18 were `ablation-final`, `baseline-d-final`, `classifier-final`, and `preproc-ablation-final`. `phase3-final`, `q-inner-final`, and `q-manifest-frozen` are local-only tags; cite commits `62ce804`, `f5010b0`, and `449c266` respectively.

## 2. Flash-economy vocabulary

| Term | Meaning in this component | Evidence |
|---|---|---|
| Sequence | A decoder accumulation step. A conventional Study-D row/column sequence has row/column flashes; an archived compact scan has two passes, not an equivalent conventional sequence. | Component 5; `prototype/server.py` |
| Flashes/character | Presented stimulus groups or keys divided by decoded characters. It must be taken from the experiment-specific trace and not inferred by multiplying a sequence count by 17. | `dead_arms/results/*.json` |
| Operating point | A specified `(tau, min_sequences, max_sequences)` plus classifier, alpha, population, and evaluator. | `results/tables/ablation_inner_m0_policy*.json` |
| Time and ITR | Component 5's variable-time model and ITR definition. Values here are reported only with their exact evaluator and policy. | [Component 5](05_language_model_prior.md) |
| Matched accuracy | Scheduler comparisons may be called matched only when absolute accuracy difference is at most .01. | `splits/scheduler_lock.json` |

Do not compare archived pre-audit Study-D replay flashes with clean-M0 OOF sequence counts or with a future adaptive-scheduler simulation.

## 3. Dynamic stopping: selection history and paper lock

The legacy sweep is **SUPERSEDED**. Its code says it evaluates held-out sessions and its threshold was selected using held-out Study-D sessions with the contaminated historical classifier. It is retained only for audit in `results/tables/dynamic_stopping_threshold_sweep.json`; it is not a valid held-out estimate and is not a reproduction target.

| Policy | Classifier / alpha | Population and selection scope | Result | Artifact | Status |
|---|---|---|---|---|---|
| Legacy selected policy, tau=.80/min=2/max=10 | Historical replay classifier; alpha not a clean-M0 policy label | Held-out Study-D sessions; threshold selection used those sessions | Historical output exists, but is contaminated by held-out-informed selection | `results/tables/dynamic_stopping_threshold_sweep.json`; `scripts/sweep_dynamic_stopping.py` | **SUPERSEDED** |
| tau=.80/min=2/max=10 | Clean M0 fixed fusion, alpha=.85 | Study-D nested run-grouped OOF; 498 characters; training-pool policy selection | Selected; 84.34% stopped accuracy, 4.912 mean sequences; 0 decision flips from default | `results/tables/ablation_inner_m0_policy.json` | **FINAL policy-selection evidence** |
| tau=.70/min=2/max=10 | Clean M0 fixed fusion, alpha=.10 | Study-D nested run-grouped OOF; 498 characters; candidate selection only | 81.93% for candidate and default; mean sequences 6.450 to 5.731; 14 changed decisions, 6 corrected, 6 regressed, 2 wrong-to-wrong | `results/tables/ablation_inner_m0_policy_alpha_0p1.json` | **CANDIDATE; not manifest-evaluated** |
| tau=.60/min=2/max=10 | Study-Q clean nested OOF LM fusion, alpha=.1 applied once, DistilGPT2 prior | Study-Q clean training sessions only; 2,592 nested outer-OOF characters; held-out Test not read | Selected replay candidate: 68.75% fused accuracy and 9.093 mean sequences versus the tau=.80 reference's 68.75% and 9.583 | `results/tables/study_q_nested_fusion.json`; `scripts/replay_study_q_policy_sweep.py` | **CANDIDATE; not applied** |

The Study-Q fusion result itself used tau=.80 at its recorded-depth protocol; tau=.60 is a training-only replay candidate, not a Study-Q Test policy. The paper/live lock is **tau=.80, min=2, max=10**. The alpha=.10 and Study-Q candidates are not applied configurations.

## 4. Archived Focus Scan / row-focus dead arms

These arms predate the leakage audit, use an untracked classifier checkpoint, and use pre-audit held-out populations. They are **DEAD** and must not be cited as current clean-M0 or live evidence.

| Arm | Mechanism | Population / result | Evidence | Status |
|---|---|---|---|---|
| Standard interleaved row/column replay | Fixed tau=.85, min=2, cap=15; conventional row/column replay | 18 sessions, 112 characters; 4.50 sequences/character; 76.5 flashes/character; 57.14% accuracy | `dead_arms/standard_interleaved_rc.py`; `dead_arms/results/standard_interleaved_rc_summary.json` | **DEAD / pre-audit** |
| Compact row-focus scan | Pass 1 flashes only rows 1–5. Pass 2 flashes keys in the best row, plus runner-up row when posterior difference is at most .08. | 18 sessions, 112 characters; 2.00 reported sequences/character; 25.89% accuracy. The JSON aggregate's 34.0 flashes/character is **erroneous/unsupported**. Directly averaging the 112 `selections[].flashes_used` values gives **14.00 flashes/character** (range 13–21). | `dead_arms/row_focus_compact_scan.py`; `dead_arms/results/focus_scan_flash_savings.json` | **DEAD / pre-audit** |
| Compact row-focus, adaptive-metadata subset | Same frozen row-focus decoder on sessions with adaptive metadata | 14 sessions, 88 characters; 1,192 total flashes; 13.55 flashes/character; 28.41% accuracy | `scripts/replay_row_focus_subset.py`; `dead_arms/results/row_focus_adaptive_subset.json` | **DEAD / pre-audit** |

The historical claim of `117.3 flashes/session at 50.6% accuracy` occurs in `scripts/sweep_dynamic_stopping.py`, but the checkpoint behind it is untracked; it is **UNVERIFIED**. It must not be combined with the 76.5-flash/57.14% standard summary; they are separate legacy summaries.

The supported failure mechanism is structural: rows 6–9 receive no evidence or flashes, so targets in those rows cannot be selected. The artifacts do **not** establish an error-compounding explanation. A local, unpushed `dead_arms/README.md` note reports 13.89 flashes/character from logged plans; it is absent from `origin/optimise`, is **UNVERIFIED from origin**, and differs from the 14.00 direct artifact calculation.

The live replay implementation is `ReplayService._decode_row_focus` and `ReplayService.replay(..., scan_mode="compact")` in `prototype/server.py`; its UI counterpart is `prototype/focus.html`. Those locations are noted here only to identify the archived decoding code.

## 5. Language-guided scheduler: implementation and design requirements

`src/stimulus/scheduler.py` is a software prototype. `SchedulerState` contains only observable state: posterior, flashed codes, optional LM/RAG priors, dimensions, and optional binary stimulus masks. `LanguageGuidedScheduler.next_flash` returns the next 1-based candidate code or `None` after its local confidence stop. It has no scheduler-outcome result.

### 5.1 Implemented modes and equations

`SCHEDULER_MODES = ("uniform", "lm_guided", "lm_rag", "closed_loop")`.

For a normalized character distribution `d` over an `n_rows × n_cols` grid, conventional row/column candidate group masses are:

\[
m = [\sum_{c \in r_1} d_c,\ldots,\sum_{c \in r_R}d_c,\sum_{c \in k_1}d_c,\ldots,\sum_{c \in k_C}d_c].
\]

With `stimulus_masks`, each binary mask is a candidate group and its mass is `mask @ d`. Empty or non-binary masks are rejected. Thus Study-Q-like group flashing can use supplied masks rather than row/column groups.

The unnormalized mode weights are transcribed from `_weights`:

| Mode | Distribution whose group masses are weighted |
|---|---|
| `uniform` | weight 1 for every candidate |
| `lm_guided` | normalized `lm_prior` |
| `lm_rag` | `0.5 * (mass(lm_prior) + mass(rag_prior))` |
| `closed_loop` | normalized elementwise product `posterior * lm_prior * rag_prior` |

If `information_gain_variant=True`, each mode's group mass `p` is replaced by the idealized binary-partition entropy:

\[
H_2(p)=-p\log_2p-(1-p)\log_2(1-p).
\]

After removing already flashed codes (or restoring all codes if all have been flashed), `_weights` normalizes to `w_i`. `next_flash` uses exactly:

\[
P(i)=\frac{\lambda}{N}+(1-\lambda)w_i,
\]

then renormalizes and samples a candidate. Here `lambda = exploration_floor` and `N` is the count of currently eligible candidate codes. This is the code-level counterpart of FR-LGF-05's proposed mixture; it mixes **group weights**, not direct character probabilities.

### 5.2 Requirement mapping

The supplied design-intent file is a proposal, not a result source. FR-LGF-09 is absent from that file; no FR-LGF-09 is invented below.

| Requirement | Implemented where | Evidence | Status |
|---|---|---|---|
| FR-LGF-01: language-guided selection | `LanguageGuidedScheduler._weights`, modes `lm_guided`, `lm_rag` | LM/RAG priors become candidate-group weights | Implemented prototype; no outcome result |
| FR-LGF-02: character probabilities | `SchedulerState.lm_prior`; `_distribution` validates a length-`R×C` nonnegative vector | Character prior is normalized before grouping | Interface implemented; upstream predictor is Component 5 |
| FR-LGF-03: word-level prior | `SchedulerState.lm_prior` / `rag_prior` interface | Scheduler consumes only resulting character distributions | Upstream LM/RAG conversion belongs to Components 5–6; scheduler evidence NOT RUN |
| FR-LGF-04: intelligent scheduling | `_group_masses`, `_weights`, `next_flash` | Weighted sampling without repeated-code preference until candidate reset | Implemented prototype; no comparative experiment |
| FR-LGF-05: exploration/safety | `next_flash` exploration-floor mixture | Formula above; λ grid locked in `scheduler_lock.json` | Implemented; λ selection NOT RUN |
| FR-LGF-06: EEG-driven adaptation | `closed_loop` uses `state.posterior * lm_prior * rag_prior` | State permits externally updated posterior | Prototype interface only; closed-loop outcome NOT RUN |
| FR-LGF-07: confidence-aware reduction | `next_flash` returns `None` if `len(flashed_codes) >= min_flashes` and `max(posterior) >= confidence_stop` | Code; Component 5 defines the paper stopping policy | Implemented local stop; scheduler outcome NOT RUN |
| FR-LGF-08: RAG-guided scheduling | `rag_prior`; `lm_rag` and `closed_loop` modes | Code uses RAG prior in weights | Prototype interface only; personalization scheduling NOT RUN |
| FR-LGF-09 | Absent from supplied requirements file | No source requirement | **ABSENT; not invented** |
| FR-LGF-10: experimental validity / no target leakage | `SchedulerState`; `next_flash`; `tests/test_scheduler_leakage.py` | State has no `target` field; `next_flash(self, state)` rejects an arbitrary mapping containing a target | Target-blind API test passes; timing/markers/labels are not scheduler-validated here |
| FR-LGF-11: controlled ablation | Four modes are coded; four-mode protocol is locked | `scheduler_lock.json` requires uniform, LM, LM+RAG, closed-loop comparison | Design/lock implemented; ablation NOT RUN after G3 no-go |

### 5.3 Target-blindness test

`tests/test_scheduler_leakage.py` checks that `SchedulerState` has no `target` field, `next_flash` takes only `self, state`, an arbitrary target-bearing mapping raises `TypeError`, modes return an unflashed code, group masks work, and the information-gain variant favors a balanced partition. It was run in the user's local `.venv` with:

```bash
.venv/bin/python -m pytest -q tests/test_scheduler_leakage.py
```

Result: **8 passed in 0.63s**. This is a run in the user's environment, not a committed result artifact.

## 6. Preregistered scheduler simulation protocol

`splits/scheduler_lock.json` has SHA-256 `eac9be209c4447c3854a38ae07a21e8e93a667b26bbe8287df460500f819b61c`, status `preregistered_before_scheduler_simulation`, and was created from `f5010b0b5f626753fe22b75adc774ff6c9d9b7a7`.

| Locked field | Protocol | Run state |
|---|---|---|
| Inputs | Clean D OOF `classifier_nested_calibration_scores.npz`; clean Q OOF `study_q_nested_fusion_scores.npz`; Q Test gate is one-shot validation only | Inputs used for validation only |
| Modes | uniform, lm_guided, lm_rag, closed_loop | Four-mode outcome simulation **NOT RUN** |
| λ grid | 0, .05, .1, .2, .3, .5; choose only from clean D/Q training-pool OOF simulation | Selection **NOT RUN** |
| Information-gain ablation | Per-group `H2(p(group))`, then same exploration floor; no target input | Coded; comparative result **NOT RUN** |
| Target mix | 25% targets sampled from bottom quartile of fixed LM prior; remainder sampled from prior; target drawn before scheduler and private | **NOT RUN** |
| Metrics | Flashes/character at matched accuracy, variable-time ITR, false lock-in, low-probability recovery, non-uniform-density calibration error | **NOT RUN** |
| Thresholds | Matched only if absolute accuracy difference <= .01; false-lock-in bound .05 | No outcome exists |
| Calibration | Compare uniform-trained calibration with group-density-stratified calibration; no outcome recalibration | **NOT RUN** |
| Leakage | Target absent from state/call signatures; rerun final spy test | Spy test run locally: 8 passed |
| Decision | Choose adaptive row/column repetition versus group subsets only from lock; if G3 fails, uniform live flashing | G3 failed; uniform retained |

The lock’s `run_status` is `simulator_validation_failed_D_reference_scheduler_future_work` and its `gate_decision.g3` is no-go.

## 7. EEG simulator and G3 validation

`scripts/validate_eeg_simulator.py` uses only clean training-pool nested OOF flash logits. For each held-out subject it pools empirical target/non-target logit distributions from all other subjects, samples i.i.d. flash logits, and repeats 20 times with seed `20261008`. It computes observed and simulated accuracy at depths 1–10 and fixed tau=.80 stopping accuracy. It writes `results/tables/eeg_simulator_loso_validation.json`; the script has **no CLI**.

| Study | Population / model input | Depth-curve MAE / maximum pointwise error | Curve rule | Known-result test | Result |
|---|---|---:|---|---|---|
| D | 17 subjects, 498 characters; clean training-pool nested OOF flash logits | .0792 / .1359 | Pass: <=.10 MAE and <=.15 maximum; maximum passes narrowly | Simulator's own OOF: observed .8193 vs simulated .9370, overshoot .1178. Preregistered reference: .9370 vs manifest Rung 1 .7581, error .17895 > .05 | Curve **passes narrowly on maximum**; own-OOF overshoot and known-result check show over-optimism; known-result **fails** |
| Q | 36 subjects, 2,592 characters; clean training-pool nested OOF flash logits | .0282 / .0511 | Pass: <=.10 MAE and <=.15 maximum | Simulated .7167 vs locked reference .6875; error .02919 <= .05 | Curve **passes**; known-result **passes** |

The curve passes and D known-result failure are separate checks, so they are consistent with the G3 no-go. The preregistered D reference is the manifest Rung 1 value (.7581), a different population from clean OOF because the manifest includes Dyn/DynBigram and uses a different alpha. This mismatch does not explain the failure away: the simulator also overestimates its own clean D OOF accuracy by 11.78 percentage points (.9370 versus .8193). The simulation is labeled “simulated; not measured EEG.” Its stated limitation is i.i.d. flash-logit sampling: it omits within-character temporal dependence, adaptation, fatigue, and nonstationary artifacts.

## 8. Experiments and evidence register

| ID | Question | Method / population | Result | Artifact / commit | Branch | Status |
|---|---|---|---|---|---|---|
| S1 | What did legacy dynamic stopping report? | Historical held-out Study-D replay | Threshold selection used held-out data | `dynamic_stopping_threshold_sweep.json` | `optimise` | **SUPERSEDED** |
| S2 | Which D clean-OOF policy preserves pooled accuracy? | M0, alpha=.85, 498 D nested OOF characters | tau=.80/min=2/max=10; zero flips | `ablation_inner_m0_policy.json` | `optimise` | **FINAL selection evidence** |
| S3 | Can alpha=.1 shorten clean OOF decoding? | M0, alpha=.1, 498 D nested OOF characters | 6.450 to 5.731 sequences at 81.93%; not manifest-evaluated | `ablation_inner_m0_policy_alpha_0p1.json` | `optimise` | **CANDIDATE** |
| S4 | Which Q OOF replay candidate uses fewest sequences? | Alpha=.1, DistilGPT2 prior; 2,592 Q clean nested OOF characters | tau=.60 candidate; no Test read | `study_q_nested_fusion.json` | `optimise` | **CANDIDATE** |
| S5 | Does Focus Scan save flashes safely? | Pre-audit Study-D archived replays | Low accuracy; structural dead rows; flash aggregate corrected above | `dead_arms/results/*.json` | `optimise` | **DEAD** |
| S6 | Does simulator reproduce clean OOF behavior? | D/Q LOSO, 20 replicates | Curves pass; D known result fails | `eeg_simulator_loso_validation.json`; `62ce804` | `optimise` | **FINAL G3 no-go evidence** |
| S7 | Do four schedulers achieve matched-accuracy savings? | Locked future simulation | Not performed after D validation failure | `scheduler_lock.json` | `optimise` | **NOT RUN / FUTURE WORK** |

## 9. Claims boundary

### Can be claimed

- Clean OOF policy selection supports paper/live tau=.80/min=2/max=10, with the exact scope in Section 3.
- The target-blind scheduler API, group masks, four modes, exploration floor, and idealized information-gain variant are implemented as a prototype.
- Both simulator depth curves pass their specified curve tolerance, while the D known-result check fails and G3 is no-go.
- Uniform flashing is the locked live policy.

### Cannot be claimed

- No matched-accuracy adaptive-scheduler flash saving exists.
- No scheduler ITR, false-lock-in, low-probability-target recovery, or non-uniform-density calibration result exists.
- No live acquisition or participant scheduler result exists.
- No Focus Scan or legacy dynamic-stopping number is a clean-M0 current held-out estimate.
- The historical row-focus checkpoint and `117.3 / 50.6%` claim are **UNVERIFIED**.

## 10. Final locked configuration

| Item | Lock |
|---|---|
| Paper/live stopping policy | tau=.80, min=2, max=10 |
| Scheduler lock | `splits/scheduler_lock.json`, SHA-256 `eac9be209c4447c3854a38ae07a21e8e93a667b26bbe8287df460500f819b61c` |
| G3 | No-go: D known-result error .17895 exceeds .05 |
| Live flashing | Uniform |
| Adaptive scheduler | Future work pending corrected simulator and preregistered successful simulation |

## 11. Limitations and open items

- Alpha mismatch: the clean D final selection artifact uses alpha=.85, while the alpha=.1 result is a non-manifest candidate; they cannot be pooled.
- Study-Q tau=.60 is a clean nested-OOF replay candidate only. It was not applied to Study-Q Test, whose recorded-depth fusion result used tau=.80.
- There is no live acquisition and no scheduler outcome evaluation.
- The simulator ignores temporal dependence, adaptation, fatigue, and nonstationary artifacts.
- FR-LGF-09 is absent from the supplied requirements file.
- The 34.0 compact aggregate is unsupported; the local-only 13.89 note is unverified from origin and differs from the artifact-derived 14.00.
- Legacy README-style historical claims, including the untracked row-focus checkpoint result, are stale or unverified unless independently recorded above.

## 12. Reproduction and inspection

| Script / module | Verified invocation or interface | Boundary |
|---|---|---|
| `scripts/replay_study_q_policy_sweep.py` | `python scripts/replay_study_q_policy_sweep.py [--scores PATH] [--result PATH]` | Training-only clean Q OOF paths by default; rerunning overwrites the result path. |
| `scripts/sweep_dynamic_stopping.py` | `python scripts/sweep_dynamic_stopping.py --threshold .80 --min-sequences 2 --max-sequences 10 --coverage-min-sequences 1` | **Do not rerun for selection**: historical held-out evaluator, superseded. `--archive-current` and `--plot-only` are also parser-confirmed. |
| `scripts/validate_eeg_simulator.py` | No CLI; run as `python scripts/validate_eeg_simulator.py` | Writes the simulator JSON; it depends on uncommitted external cache inputs. Existing result is the evidence record; do not alter the lock after observing it. |
| `scripts/replay_row_focus_subset.py` | No CLI; run as `python scripts/replay_row_focus_subset.py` | Rewrites a pre-audit dead-arm artifact; not for model selection. |
| `src/stimulus/scheduler.py` | Import `LanguageGuidedScheduler` and `SchedulerState`; no CLI | Prototype module; no outcome evaluator. |
| `tests/test_scheduler_leakage.py` | `.venv/bin/python -m pytest -q tests/test_scheduler_leakage.py` | Local software test only; observed result was 8 passed. |

One-shot held-out evaluators are not routine reproduction commands.

## 13. Handoff block

```text
1. Paper/live stopping policy: tau=.80, min=2, max=10; Component 5 owns its rule and ITR definition.
2. D alpha=.85 clean OOF selected this policy on 498 characters with zero decision flips.
3. D alpha=.1 tau=.70 and Q tau=.60 are training-only replay candidates, not applied paper/live policies.
4. Legacy stopping and Focus Scan results are pre-audit and superseded/dead.
5. Compact Focus Scan structurally excludes rows 6–9; the artifact-derived per-selection mean is 14.00 flashes/character.
6. Scheduler prototype has uniform, LM, LM+RAG, and closed-loop modes plus group masks and exploration floor.
7. Target-blind API test passed locally (8 tests); this is not a committed experiment artifact.
8. Scheduler lock SHA-256 is eac9be209c4447c3854a38ae07a21e8e93a667b26bbe8287df460500f819b61c.
9. Simulator curves pass for D/Q, but D known-result validation fails (.17895 > .05); G3 is no-go.
10. Live flashing remains uniform; no adaptive-scheduler savings, ITR, false-lock, or low-probability-recovery outcome exists.
```
