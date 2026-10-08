# Phase 1–3 close-out and Phase 4 decision

## Phase 1: Study Q

- Frozen evaluation: `splits/study_q_manifest.json`, 180 SE003 runs and 36 subjects. Test labels are in `data/evaluation/ground_truth_vault.json`; training code uses the isolated Train registry.
- Pre-Test analysis and protocol are committed/tagged as `q-inner-final` (`f5010b0`). The prior Q classifier has no persisted model/provenance, so its training runs, preprocessing, calibration, and tuning could not be audited; all earlier Q results are exploratory.
- The one permitted SE003 replay used clean M0 trained on 360 SE001/SE002 runs and the frozen temperature/prior. It scored all 1,296 characters at their recorded depth. Fused accuracy was 59.41% versus 55.86% EEG-only (+3.55 points; subject bootstrap 95% CI +2.47 to +4.78 points; Holm p=3.95e-5). There were 27 positive subjects, 9 ties, and no negative subjects.
- The depth distribution was 36 at depth 4, 108 at depth 5, 216 at depth 6, 216 at depth 7, 108 at depth 8, 216 at depth 9, and 396 at depth 10. Depth-curve results are explicitly biased by early stopping and unequal recorded depth. Stopping-policy accuracy, flashes/character, and ITR are dropped.
- By condition, fused accuracy was 65.66% for ColorIntensification, 56.71% for Grey-to-Color, and 56.62% for Grey-to-White. The fused gain stayed positive in the 4–6, 7–8, and 9–10 depth buckets. Relative to 68.71% full-depth inner OOF, the Test fused result is 9.30 points lower, just inside the preregistered 10-point session-shift flag. G1 passes narrowly; retain the shift caveat.
- Full result: `results/tables/study_q_test_once.json`; gate summary: `results/tables/study_q_phase1_gate.json`.

## D+Q inner-OOF summary

The subject-level fixed-policy fusion effect is summarized with a DerSimonian–Laird random-effects calculation across two studies: +5.46 percentage points, normal 95% interval −0.48 to +11.41. This is descriptive only; between-study variance is unstable at two studies, and D row/column flashing differs from Q group flashing. See `results/tables/dq_inner_random_effects.json`.

## Phase 2: RAG

G2 is **no-go for a natural-text personalization claim**. Q phrases are typed grid-key labels and do not represent natural-language user history; Study D task prompts also do not qualify as prospective user-authored history. Participant access is pending, so there is no qualifying per-user corpus for chronological train/evaluation splits, language-level metrics, learning curves, topic switching, or robustness against real user text.

The requirements are preregistered in `splits/rag_lock.json`. The generic `RAGPredictor.update(decoded_text)` API adds completed text to an in-memory session bank and does not persist it. It is not a measured gain. Stop RAG performance engineering until consented natural-text histories are available.

## Phase 3: scheduler

The scheduler accepts only observable priors/posteriors and history, supports Study Q group masks, and implements the locked idealized information-gain weighting variant. Leakage tests confirm that the target is absent from scheduler state and calls. `splits/scheduler_lock.json` freezes the four modes, λ grid, low-prior target mix, matched-accuracy rule, false-lock-in bound, and calibration sensitivity analysis before any simulation.

The EEG simulator and scheduler simulation have **not** run. Thus G3 is unevaluated. Do not select λ, claim reduced flashes, or deploy adaptive flashing from software tests alone. Keep uniform flashing for live data until the simulator and pre-registered simulation pass.

## Phase 4 go/no-go

| Arm | Gate | Decision |
|---|---|---|
| Clean Q M0 EEG and fixed LM fusion | G1 passed narrowly | Retain as a research result; validate session shift before transfer claims |
| RAG personalization | G2 not evaluable; no qualifying corpus | No-go until participant text is available |
| Adaptive scheduler | G3 not run; simulator missing | No-go for live deployment; uniform flashing remains |
| Real-time participant study | Participants pending | Software self-test/demo only |
