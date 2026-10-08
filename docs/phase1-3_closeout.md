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

## Phase 2: RAG software arm

The RAG stack is implemented, but G2 is **no-go for a natural-text personalization claim or live-session use**. Participant access is pending; D prompts and Q's repeated grid-key strings are not suitable user-authored history. The participant protocol is locked in `splits/rag_lock.json`: anonymize before import, require 700 whitespace tokens, preserve chronology, allocate 60% to bank building, 20% to gate tuning, and 20% to held-out scoring. The local preparation utility redacts email and phone patterns and keeps text under the ignored `data/rag/participant_local/` path; names and sensitive spans still require participant-reviewed redaction. No participant text was accessed.

The public-domain development smoke test used four author-separated texts, chronological splits, 24 held-out character cases per author, and 8 gate-tuning cases. Macro next-character cross-entropy was 4.463 nats for LM-only, 4.151 with a global bank, 4.245 with an author bank, and 3.763 for the gated blend; macro top-3 accuracy was 0.323, 0.448, 0.458, and 0.479 respectively. This is a capped software diagnostic with only 96 held-out cases and is **not** evidence of personalization. Per-author results include degradation for global and subject banks, and the cross-author topic switch worsened NLL from 4.739 to 5.217 nats; the bounded-degradation gate therefore fails in this diagnostic. Learning curves, cold-start updates, and OOV/special-key/backspace/empty/adversarial-bank checks run only as software diagnostics. The synthetic self-generated end-to-end dry run scored 12 next-key priors; all were finite and normalized, and its evaluation tail was not added to the bank.

`RAGPredictor.update(decoded_text)` updates only the in-memory session bank. Participant-level paired comparisons, Holm correction, and participant-bootstrap intervals cannot be computed until eligible participant samples exist. Until then RAG remains implemented but unproven, and is excluded from live-session time.

## Phase 3: EEG simulator and scheduler

The preregistered simulator used clean D/Q training-pool OOF logits only. Its leave-subject-out accuracy-vs-depth curves passed the locked curve tolerance in both studies (D MAE 0.079, max point error 0.136; Q MAE 0.028, max 0.051). However, the known-result check failed for D: simulated fixed-policy accuracy was 0.937 versus the locked 0.7581 reference, an absolute error of 0.179 against the 0.05 limit. Q passed (0.7167 versus 0.6875, error 0.0292). The simulator therefore fails validation as a cross-study foundation. It samples i.i.d. flash logits and ignores within-character temporal dependence, adaptation, fatigue, and nonstationary artifacts.

G3 is decided **no-go / future work**. In accordance with `splits/scheduler_lock.json`, no λ was selected and no four-scheduler outcome simulation or calibration-sensitivity result was run after the D validation failure. The target-blind scheduler module remains a software prototype; the final spy test passes. Do not claim matched-accuracy savings, ITR, false-lock performance, or unlikely-target recovery. Live data collection keeps uniform flashing. The validation record is `results/tables/eeg_simulator_loso_validation.json`.

## Phase 4 go/no-go

| Arm | Gate | Decision |
|---|---|---|
| Clean Q M0 EEG and fixed LM fusion | G1 passed narrowly | Retain as a research result; validate session shift before transfer claims |
| RAG personalization | G2 no-go; small public-domain smoke only, participant study pending | Exclude from live sessions until participant gate passes |
| EEG simulator | D known-result check fails; Q and depth curves pass | Do not use for scheduler claims |
| Adaptive scheduler | G3 no-go after simulator failure; scheduler simulation not run | Future work; uniform flashing remains |
| Real-time participant study | Participants pending | Software self-test/demo only |

## Phase 4 go/no-go note

EEG-only and fixed LM fusion remain the only arms with Study D/Q evaluation evidence. RAG is conditional on participant-text G2; adaptive scheduling is no-go until a corrected simulator passes and a preregistered simulation earns G3. The current real-time plan therefore uses uniform flashing with the existing EEG-only/LM-fusion comparison.
