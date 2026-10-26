# 02 — Preprocessing

Scope: EDF decoding, event/label construction, epoching, and the classifier-cache interface. Classifier fitting and decoding are downstream interfaces only. Code references below describe checked-in code; where paper freeze or run evidence is absent, it is marked `UNVERIFIED` or `NOT RUN`.

## 1. Purpose and position in pipeline

**Input:** bigP3BCI EDF recordings. The parser requires channels named with an `EEG_` prefix, plus `StimulusBegin`, `StimulusType`, `StimulusCode`, and `CurrentTarget` trigger channels. Adaptive paths also require `SelectedTarget`. Cell channels encode `label_row_col` (parsing splits from the right, allowing punctuation labels).

**Primary output:** an MNE `Epochs` object, spelled-text string, and grid metadata from `parse_bigp3bci_edf`. The batch command saves each epoch object as `data/processed/<Study>/<stem>-epo.fif`, plus JSON registries for grid and ground truth. Epoch data are `(n_epochs_kept, n_EEG_channels, n_time_samples)`; labels are `epochs.events[:, 2]` with `0=non-target`, `1=target`. Metadata carries `stimulus_code` (the flashed row/column or group-flash code) and `stimulus_type` (same binary event label); adaptive data additionally carry `char_index` (0-based) and `sequence_in_char` (1-based). Group-flash data also carry `lit_mask` (row-major `0`/`1` cell string).

**Downstream interface:** an optional cache builder converts FIF epochs to classifier HDF5. The parser itself emits FIF, not HDF5; cache schemas and classifier-side transforms belong to the classifier component documentation.

Event timing, target labels, cell identity, and character/sequence boundaries determine the downstream classifier’s training examples and grouping. The locked Study D manifest evaluation uses 1–12 Hz. A fourth M1 SWLDA arm at 1–12 Hz, baseline off, is directly comparable to the 0.1–30 Hz/baseline-off arm: on this clean split, 1–12 Hz scored higher AUC and character accuracy, with paired run-bootstrap intervals above zero. This supports the locked choice indirectly on this protocol; M1 is not the paper-reported final model. See P9 in §4.

## 2. Sequential processing steps

The primary implementation is `src/data/batch_preprocess.py:parse_bigp3bci_edf`, followed by its `__main__` batch loop.

| # | Operation and purpose | Parameters / code location | Shape and failure behavior |
|---|---|---|---|
| 1 | Read EDF fully into MNE `Raw`; preserve original sample clock. | `mne.io.read_raw_edf(..., preload=True)` in `parse_bigp3bci_edf`; no parser-level resampling. | `(channels, samples)`; EDF read errors propagate to batch handler, which logs file and traceback in `batch_preprocess_failures.log`. |
| 2 | Identify EEG and grid-cell channels; rename `EEG_<name>` to `<name>`; attach standard 10–20 montage where positions exist. Grid map comes from the final two underscore-separated numeric fields. | Montage `standard_1020`; `on_missing='ignore'`; parser in `parse_bigp3bci_edf`. | Raw array dimensions unchanged. Missing EEG-prefixed channels raises `ValueError`; missing required triggers raises `ValueError`; adaptive file without `SelectedTarget` raises `ValueError`. |
| 3 | Bandpass EEG only; optional line-noise notch. Trigger and cell channels are not selected for this filter. | Defaults `l_freq=0.1`, `h_freq=30.0`, `notch_freq=None`; the controlled ablation can pass both cutoffs as `None` to skip bandpass. | Same raw shape. Filter errors propagate. No common average/reference transform is performed in this function. |
| 4 | Detect flash onsets from `StimulusBegin` rising edges using midpoint of channel min/max; label each event from `StimulusType > 0.5`; sample `StimulusCode` at each onset. | Threshold `(min+max)/2`, event extraction in `parse_bigp3bci_edf`. | `events: (n_events,3)`, columns `[sample,0,binary_label]`; `stimulus_codes_per_event: (n_events,)`. No edges raises `ValueError`. |
| 5 | Infer grid dimensions, map grid indices to labels, and detect group-flash conditions. Group flash is inferred when max observed `StimulusCode > n_rows+n_cols`; cell-channel activity supplies per-event `lit_mask`. | Grid parse and `extract_lit_masks(..., window=8)`; row-major cell order; `n_rows=max(row)`, `n_cols=max(col)`, label index `(row−1)*n_cols+col`. | `lit_mask: (n_events,)` strings of length number of cells. Group-flash validation raises if cell count differs from `n_rows*n_cols`, lit counts are zero/nonconstant, or group size does not tile the grid. `StimulusCode` remains metadata; group-flash cell membership is taken from cell channels. |
| 6 | Extract spelled text and character/sequence boundaries. Fixed RC runs vote over chunks of target-flash `CurrentTarget` codes. Adaptive files use nonzero `SelectedTarget` rising pulses; each pulse represents one finalized selection, including repeated adjacent symbols. Non-adaptive group-flash files without selection pulses use runs of target codes for registry text. | `sequences_per_selection=20` default; fixed-block agreement warning below 0.9. Adaptive sequence size is inferred as `n_rows+n_cols` for row/column or `(grid_cells/lit_cells_per_flash)` for group-flash. `_annotate_adaptive_event_indices` numbers sequences within each selected character. | Output text from grid-code lookup; unknown code rendered `[code]`. Non-divisible fixed target count warns and omits incomplete remainder (`n_blocks=floor`). Adaptive events outside any pulse-defined character are `char_index=-1`, `sequence_in_char=-1`. Group-flash adaptive sequence coverage below 95%, or no complete sequence, raises `ValueError`. |
| 7 | Epoch each event, baseline-correct, and optionally reject/decimate; preserve metadata after MNE drops edge epochs by indexing with `epochs.selection`. | Defaults `tmin=-0.1 s`, `tmax=0.8 s`, `baseline_correction=True` (baseline `(tmin,0)`), `reject_threshold=None`, `decim=1`; all set in `parse_bigp3bci_edf`. | `X: (n_kept, n_EEG, n_times)`. Binary class IDs target=1, non-target=0. `n_times` follows sampling rate and MNE inclusive epoch endpoints; edge events may be dropped. Optional rejection drops epochs above EEG peak-to-peak threshold. |
| 8 | Save FIF and registries; keep training labels separate from evaluation labels and check static grid consistency. | Batch loop in `__main__`; Study D RC/Train labels go to `ground_truth_registry.json` only if not in held-out manifest; other labels go to `data/evaluation/ground_truth_vault.json` (chmod `0600`). | Writes one `*-epo.fif`; `grid_layout.json`; ground-truth registry/vault; adaptive quality flags CSV as applicable. Exceptions are logged and processing continues. Grid mismatch raises for that file. Existing FIF files are skipped. |
| 9 | Optional conversion to the downstream classifier cache. | `scripts/build_classifier_cache.py` or the Study Q-specific cache builder. | FIF remains the preprocessing output. Cache datasets, metadata, and HDF5 attributes are specified in classifier-component documentation. |

**Sample-rate handling:** parser retains native EDF `sfreq`; it does not resample. The locked Study D manifest evaluator parses raw EDF directly at 1–12 Hz with parser decim 1, then applies the classifier window and decimation in the model feature path. Study Q cache construction applies 1–12 Hz and decim 2 while parsing. Batch FIF defaults (0.1–30 Hz, baseline on) are a separate path; conversion from existing FIF to HDF5 does not re-filter epochs.

## 3. Data and event flow

```text
EDF Raw (EEG_*, cell-label_row_col, trigger channels)
  -> rename EEG channels; set montage; select a pipeline-specific EEG bandpass
  -> StimulusBegin rising edges -----------------------------+
       | sample StimulusType -> event y: 0 non-target / 1 target
       | sample StimulusCode -> stimulus_code metadata       |
       |                                                     |
       +-> cell channels -> row-major lit_mask (group flash) |
       +-> CurrentTarget target flashes -> fixed blocks / majority vote
       +-> SelectedTarget 0-to-nonzero pulses -> adaptive character boundaries
  -> MNE epoch [-0.1, 0.8] s; baseline; optional reject/decim
       -> epochs.events[:,2] binary y; metadata follows epochs.selection
  -> batch parser path: default 0.1–30 Hz, baseline on -> FIF + registries
       -> optional FIF-to-HDF5 cache conversion (does not re-filter)
  -> locked Study D manifest path: raw EDF directly at 1–12 Hz, baseline off
       -> pre-evaluation M0 vs M3a comparison; paper decision retains M0; feature transform uses decim 2
  -> Study Q cache path: raw EDF directly at 1–12 Hz, baseline off, decim 2 -> HDF5
```

**Real EDF example:** `D_15_SE001_Dyn_Test01.edf` (Study D). The checked-in file audit records 256 Hz, 32 EEG channels, a 9×8 grid, and 234 flash events. Parsing this real EDF with the `optimise` branch `.venv` and `parse_bigp3bci_edf` yielded epochs shape `(234,32,232)`, labels `{0:204, 1:30}`, spelled text `CANCEL`, grid `(9,8)`, and metadata columns `stimulus_code`, `stimulus_type`, `char_index`, `sequence_in_char`. The raw event array therefore has shape `(234,3)`; all 234 events were retained in this file. This file’s event ratio is 30 targets to 204 non-targets (1:6.8). The clean Study D classifier pool contains 9,960 targets and 74,700 non-targets across 84,660 epochs (1:7.5). This example uses no synthetic data.

## 4. Experiments and evidence

| ID | Question | Method | Result | Evidence file/commit | Branch | Status |
|---|---|---|---|---|---|---|
| P1 | Can fixed target-flash blocks recover RC calibration text? | 20 consecutive target `CurrentTarget` codes per character; majority vote. | Current implementation; low agreement warns below 90%. The historical “confirmed empirically” statement has no dedicated comparison artifact located. | `src/data/batch_preprocess.py`; `42a4e3d` | `optimise` / restored common code | FINAL in code; empirical evidence UNVERIFIED |
| P2 | Can adaptive segmentation preserve repeated symbols? | Detect each nonzero `SelectedTarget` rising pulse; map pulse code to grid label. | Code explicitly handles adjacent identical selected codes (example `BUTTON`); no quantitative before/after comparison artifact located. | `src/data/batch_preprocess.py`; `5af0c7a` | `optimise` | FINAL in code; quantitative comparison NOT RUN |
| P3 | Are all Study D layouts identical? | Normalize parsed coordinate tuples to JSON-style lists before comparing. | Source comment records 205/205 prior false mismatches were tuple/list type-only; layout check remains. | `src/data/batch_preprocess.py`; `42a4e3d` | `optimise` | FINAL in code; reported count is source-comment evidence |
| P4 | Did preprocessing complete for all 307 Study D EDFs? | Compared canonical EDF stems against saved FIF stems, output timestamps, append-only failure log, and grid audit. | **Final 2026-10-06 pass:** 307/307 canonical Study D EDFs have matching `*-epo.fif` outputs created that date; the append-only failure log has no Study D entries that date. The batch code appends one failure record per caught file exception, so this supports 0 logged failures for that pass. Its grid mismatch check raises inside the same handler; the raw grid audit also reports all 307 as 9×8 with 0 missing cells. The log separately records 410 earlier Study D exceptions on 2026-08-18 across 205 unique EDFs (`UnboundLocalError` for `selected_target_ch`); the full processing history did not have zero failures. | `data/processed/batch_preprocess_failures.log`; `data/processed/StudyD/*-epo.fif` inventory; `data/processed/grid_layout.json`; `results/bigp3bci_studyd_e_findings/bigp3bci_studyd_e_findings.md`; `src/data/batch_preprocess.py` | `optimise` | FINAL for 2026-10-06 pass; earlier failed attempts recorded |
| P5 | Did 2D RP+CWT CNN input benefit from `log1p` + instance normalization? | CNN forward applies `log1p` and `InstanceNorm2d(affine=False)` to CWT magnitudes. | Branch comments report training instability as motivation; no ablation isolating normalization effect found. | `src/models/cnn_classifier.py`; commits `5889290`, `911c89a`; `INVESTIGATION_SUMMARY.md` | `CNN` | DEAD / comparison NOT RUN |
| P6 | Did RP+CWT/CNN outperform SWLDA? | Archived exploratory RP+CWT feature/classifier comparison. | The reported 22.06% CNN vs 32.66% SWLDA baseline is **SUPERSEDED**: the SWLDA number predates the leakage audit and is not a valid clean comparator. Do not cite this as a preprocessing effect or current classifier comparison. | `INVESTIGATION_SUMMARY.md`; `results/tables/classifier_protocol_audit.json`; `README.md` §3.11 | `CNN` | SUPERSEDED / DEAD |
| P7 | What changes when the Study D 0.1–30 Hz bandpass is omitted? | Paired clean-pool M1 tuned SWLDA; 3-fold GroupKFold by run, 3,000 fit-sample cap, baseline on in both arms. Run-cluster paired bootstrap: 10,000 replicates, seed 0, percentile 95% CI. | Filtered minus no bandpass: **+0.24 AUC pp (95% CI −0.40 to +0.93), −3.01 character pp (95% CI −7.23 to +1.20)**; 84,660 epochs, 498 characters, 83 runs. Both intervals span zero: **no detectable run-level difference** in this probe; this does not establish equivalence. | `results/tables/preprocessing_ablation_20261026.json`; bootstrap `results/tables/preproc_ablation_filter_bootstrap_p7.json`; paired scores `preprocessing_ablation_filter0130_baseline_on_scores.npz`, `preprocessing_ablation_unfiltered_baseline_on_scores.npz` | `optimise` | FINAL; paired descriptive OOF comparison |
| P8 | What changes when baseline correction is disabled? | Paired clean-pool M1 tuned SWLDA; 0.1–30 Hz bandpass in both arms. Run-cluster paired bootstrap: 10,000 replicates, seed 0, percentile 95% CI. | Baseline on minus off: **+0.30 AUC pp (95% CI −0.35 to +0.98), −2.01 character pp (95% CI −6.63 to +2.81)**; 84,660 epochs, 498 characters, 83 runs. Both intervals span zero: **no detectable run-level difference** in this probe; this does not establish equivalence. | `results/tables/preprocessing_ablation_20261026.json`; bootstrap `results/tables/preproc_ablation_filter_bootstrap_p8.json`; paired scores `preprocessing_ablation_filter0130_baseline_on_scores.npz`, `preprocessing_ablation_filter0130_baseline_off_scores.npz` | `optimise` | FINAL; paired descriptive OOF comparison |
| P9 | At the locked baseline-off setting, does 1–12 Hz outperform 0.1–30 Hz under the same M1 protocol? | Added fourth arm on the same 83 clean runs, using the 1–12 Hz/baseline-off HDF5 and the same M1 tuned SWLDA, 3-fold run-grouped CV, config (0.8 s window, decim 2, all channels), and 3,000 fit-sample cap. Metadata arrays match the earlier P7/P8 caches exactly. Paired run-cluster bootstrap: 10,000 replicates, seed 0, percentile 95% CI. | 1–12 Hz: epoch AUC **0.7599**, 498-character accuracy **67.07%**. Against P8’s 0.1–30 Hz/baseline-off result (0.7350 AUC, 62.05%): **+2.49 AUC pp (95% CI +1.94 to +3.06), +5.02 character pp (95% CI +0.80 to +9.24)**. Intervals exclude zero for this M1 probe; no model-selection or multiple-comparison correction is claimed. This supports 1–12 Hz indirectly, not as a result for the paper-final M0. | `results/tables/preproc_ablation_filter1_12_baseline_off.json`; bootstrap `results/tables/preproc_ablation_filter_bootstrap.json`; aggregate `results/tables/preprocessing_ablation_20261026.json`; candidate cache SHA-256 `7615c89e80b266d70403180d9f3880611d8201dd3c7e78384c912212b4f3093e` | `optimise` | FINAL for this probe; descriptive OOF comparison |

All four caches have identical `y`, `run_id`, `subject`, `seq_idx`, `char_idx`, and `stim_code` arrays; they share the held-out manifest hash and contain 83 training runs. The paired bootstrap artifacts preserve those run pairings. `results/tables/preprocessing_ablation_20261026.json` records the original cache/score/config/code hashes and P9 metadata; the P7/P8 bootstrap JSON files record both score hashes and interval settings.

`classifier_phase2_screens.json` also contains a `bp_1_12` screen, but it uses M2 shrinkage LDA and baseline correction on. It does not isolate the requested M1/baseline-off setting; P9 is the matched arm used here.

## 5. Dead arms and superseded choices

| Choice | What was tried | Why superseded / dead | Evidence | Archive |
|---|---|---|---|---|
| Fixed 20-target-flash segmentation for adaptive Dyn/DynBigram | Earlier parser grouped target flashes in fixed blocks. | Adaptive stopping changes the number of flashes per character; fixed blocks can mislabel text and merge repeated selections. Current code uses `SelectedTarget` pulses for adaptive paths. | Earlier warning retained in `src/preprocessing/epoching.py`; corrected path in `src/data/batch_preprocess.py`, commit `5af0c7a`. | Git history; older implementation at `src/preprocessing/epoching.py` |
| CurrentTarget transitions alone for repeated adaptive characters | Segment text when target code changes. | Adjacent repeated selected characters have the same code and no transition; pulse events preserve each finalized selection. | Current parser comments and `README.md` repeated-letter note. Quantitative comparison `NOT RUN`. | Superseded in parser; not a separate archive |
| RP/CWT 2D features and CNN | Selected eight channels; uniform-index time reduction to 64 samples; Gaussian recurrence plots and Morlet CWT; CWT magnitude normalization in CNN forward. | Exploratory classifier arm was stopped. The legacy CNN-vs-SWLDA headline uses a superseded SWLDA baseline; no standalone benefit from CWT normalization was measured. | `src/preprocessing/feature_transforms.py`, `scripts/compute_2d_features.py`, `src/models/cnn_classifier.py` at `911c89a`; `INVESTIGATION_SUMMARY.md`. | Git branch `CNN`; commit `911c89a` (“CNN Done”), later commit `38d47fb` (“dead end”) |
| Historical tuple-vs-list grid comparison | Compared tuples returned by parsing against lists loaded from JSON. | Serialization changed coordinate container type, causing false mismatches. Normalize coordinates before comparison. | Source comment says 205/205 false positives; current batch parser normalizes to lists. | Fix in `src/data/batch_preprocess.py` |

No other preprocessing dead arm is established by checked-in `dead_arms/README.md`; that README covers stimulus-selection methods, which are outside this component.

## 6. Final locked configuration

| Item | Current code/default | Frozen paper reference |
|---|---|---|
| Main parser defaults | 0.1–30 Hz bandpass; no notch; epoch −0.1 to 0.8 s; baseline on; no rejection; decim 1 | Exact defaults at the classifier-run parser blob `c241f51ca98401bbd0a8aed932677f95a4826eed`; the current workspace adds an optional no-bandpass experiment path, leaving defaults unchanged. |
| Study D separate clean-reference cache | `data/cache/classifier_epochs.h5`, 0.1–30 Hz, baseline-corrected epochs; used for clean-reference M0/OOF audit, not the finalist manifest evaluation | Cache SHA-256 `8811ec22ee33a0053e29e02aa9aa862855a1f6514e5d1b22fd5b5332dd2ffa81` in `splits/baseline_d_lock.json`; 83 clean runs, 84,660 epochs. This is a cache configuration; M0 is also a model ID and that ID alone does not specify filter settings. |
| Study D locked manifest comparison | **Effective filter 1–12 Hz** for M0 and M3a; epoch −0.1…0.8 s, baseline off, all locked channels. Parser decim 1; classifier feature path uses 0…0.8 s and decim 2. | `splits/finalist_lock.json` is pre-evaluation: M3a candidate, M0 baseline. `scripts/evaluate_classifier_manifest.py:_evaluate_run` parses raw EDF with `l_freq=1.0`, `h_freq=12.0`, `baseline_correction=False`, `decim=1`; `_score_epochs` applies window/decimation. `classifier_manifest_eval.json` records commit `e16ed2236e41480a7eb494a233d7a1e18c94ef16` and matching lock hash. The historical manifest result does not record its training-cache SHA. |
| Paper-reported final Study D classifier | **Clean M0 epoch scorer (streaming SGD); M3a was not adopted.** | `splits/ablation_lock.json` locks `classifier.id=clean_m0`; `README.md` §3 reports “do not adopt M3a; retain the clean M0 classifier.” The finalist lock records M3a only as the pre-evaluation candidate. The M0 paper-final path uses the 1–12 Hz preprocessing documented above. |
| Study D adaptive labels | `SelectedTarget` rising pulses; adaptive sequence boundary metadata | Present in the recorded run snapshot; the current parser preserves this behavior and adds only the optional no-bandpass path for P7. |
| Study Q cache | Bandpass 1–12 Hz; epoch −0.1 to 0.8 s; baseline off; decim 2; 12 flashes per group sequence | Exact Study Q builder configuration at `q-inner-final` (`f5010b0`). |
| RP/CWT + CNN | Exploratory only, `CNN` branch; not current locked classifier preprocessing | Dead arm, not frozen |

Tags inspected: `ablation-final`, `baseline-d-final`, `classifier-final`, `phase3-final`, `q-inner-final`, `q-manifest-frozen`. The classifier result JSON records its exact run commit and lock hash but not the training-cache SHA. The default local `classifier_1_12_no_baseline.h5` exists and its HDF5 attrs report 1–12 Hz and baseline off (SHA-256 `7615c89e80b266d70403180d9f3880611d8201dd3c7e78384c912212b4f3093e`); whether those exact bytes were used for the recorded evaluation is **UNVERIFIED**. At the recorded commit, parser/cache-builder blobs match `classifier-final` and `phase3-final`; the current workspace adds the P7/P8 no-bandpass option while retaining the recorded defaults. Current experiment code hashes are in `results/tables/preprocessing_ablation_20261026.json`. The Study D and Q locked classifier paths both use 1–12 Hz and baseline off; Study D manifest evaluation applies decim 2 in the model feature path, while Study Q cache construction applies decim 2 while parsing. The repository does not record a physiological rationale for the cutoff. P9 provides a matched M1 performance comparison that supports the locked 1–12 Hz setting on this clean split; P7/P8 test other filter/baseline combinations.

**Paper headline ladder (Rungs 0–4): final classifier is clean M0; effective bandpass is 1–12 Hz, baseline off.** `splits/ablation_lock.json` locks `classifier.id=clean_m0` and records a parent classifier-lock SHA that matches `splits/finalist_lock.json`; that parent lock specifies 1–12 Hz. `scripts/export_locked_clean_m0.py` fits the scorer from `classifier_1_12_no_baseline.h5` with `baseline_off.json`; the ladder runner parses evaluation EDFs at 1–12 Hz and baseline off. `ablation_ladder_clean_m0.json` records the matching ablation/classifier lock hashes. Neither the lock nor ladder JSON records the serialized M0 scorer SHA, so byte-level provenance of that exact scorer is **UNVERIFIED**; the locked/exported pipeline settings are 1–12 Hz. The 0.1–30 Hz `classifier_epochs.h5` is a separate clean-reference/OOF cache, not the ladder input. The one-shot manifest comparator and ladder use different evaluators; their accuracy outputs are not interchangeable, but both use the same locked 1–12 Hz preprocessing. P9 favors 1–12 Hz on the clean M1 split (+2.49 AUC pp, run-bootstrap 95% CI +1.94 to +3.06; +5.02 character pp, CI +0.80 to +9.24), which is indirect filter evidence and not a comparison using paper-final M0.

## 7. Validity and leakage notes

- `StimulusType` supplies per-flash target/non-target labels. `CurrentTarget` and `SelectedTarget` supply spelled-text/context labels. These are distinct: selection ground truth must not be confused with classifier event labels.
- Study D adaptive `SelectedTarget` is read from EDF to construct spelled text and character boundaries. For training, target labels remain the event-level `StimulusType`; `char_index` and `sequence_in_char` are metadata used to group events.
- Batch code keeps Study D RC/Train ground truth in the non-vault training registry only when the run is not in the held-out manifest. Other parsed text is sent to the evaluation vault with restrictive file permissions. The held-out manifest is also used by cache builders to exclude held-out run IDs.
- `scripts/build_classifier_cache.py` derives training paths from `*_Train*` epochs and removes manifest-held-out IDs; it asserts disjointness. It refuses overwrite and records the manifest hash. Study Q builder excludes the held-out SE003 session and asserts no selected Test or SE003 runs.
- The EDF parser performs per-recording filtering and baseline correction; it does not estimate normalization statistics across the full dataset. No split-fitted signal normalization appears in the parser. CNN branch `InstanceNorm2d` is per sample at inference/training and belongs to the dead arm.
- `ground_truth_vault.json` is protected by `chmod(0600)` in the batch script, but filesystem permission and access-control enforcement in a reproduced environment remain operational requirements.
- **Held-out vault rule:** do not read held-out text to tune preprocessing, train/cache normalization, select parameters, or revise the pipeline. Access evaluation labels only under the declared final evaluation protocol; preserve manifest separation and document any access. Current evaluation-vault handling is visible in code; compliance of all historical runs is **UNVERIFIED**.

## 8. Limitations and open items

- The parser’s required channel aliases and exact expected trigger voltage conventions beyond thresholding are dataset-dependent; generalized EDF compatibility is **UNVERIFIED**.
- No parser-level resampling or common-average reference occurs. These should not be described as performed preprocessing.
- Group-flash detection uses `max(StimulusCode) > n_rows+n_cols`; applicability beyond the inspected studies is **UNVERIFIED**.
- Fixed-block segmentation drops an incomplete trailing block after warning. No separate recovery path exists.
- Output epoch sample count varies with native sampling rate and MNE endpoint handling; the real Study D example above confirms `(234,32,232)` for that EDF only.
- The final artifact inventory establishes 307/307 Study D EDF-to-FIF stem matches dated 2026-10-06; the failure logger records no Study D exceptions on that date, and the raw grid report establishes 307 complete 9×8 layouts. Earlier, the append-only log records 410 Study D exceptions across 205 unique files on 2026-08-18, all reporting `UnboundLocalError` for `selected_target_ch`. The final-pass outcome is supported by outputs and logger behavior; do not summarize the full history as “zero failures.”
- P7–P9 compare only the specified Study D RC/Train clean pool and one M1 SWLDA protocol. P7/P8 intervals span zero (no detectable paired run-level difference; not an equivalence test); P9 intervals exclude zero for that M1 probe. No inferential model-selection or multiple-comparison test was run, and M1 is not paper-final M0.
- RP/CWT transform details and CNN normalization are branch-specific dead-arm documentation; do not present them as the final mainline pipeline.
- The recorded `classifier_manifest_eval.json` run commit and matching finalist-lock hash identify the Study D manifest comparison snapshot; the later paper-facing decision retains M0 and rejects M3a. Its result JSON does not record the training-cache SHA; exact input-cache provenance is **UNVERIFIED**. The 0.1–30 Hz reference-cache hash in `baseline_d_lock.json` belongs to the separate clean-reference/OOF audit, not proof of the final evaluation cache.

## 9. Reproduction commands

Run from repository root in an environment with dependencies installed. The default Study D batch path is `data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyD`; if EDFs are only under `bigP3BCI-data-backup`, the code currently has no `--raw-root` option, so the input tree must be arranged at the expected path before running. The batch parser/FIF route (0.1–30 Hz) is distinct from the locked Study D manifest evaluation route (raw EDF at 1–12 Hz).

```bash
# Regenerate missing Study D FIF epochs and registries (existing FIFs are skipped).
python -m src.data.batch_preprocess --study StudyD

# Build the default clean classifier HDF5 from the Study D FIF files.
# This preserves the FIF preprocessing (0.1–30 Hz); it does not apply a new filter.
python scripts/build_classifier_cache.py \
  --epochs data/processed/StudyD \
  --manifest splits/heldout_manifest.json \
  --output data/cache/classifier_epochs.h5

# Build the locked finalist training cache directly from raw EDF at 1–12 Hz.
python scripts/build_classifier_cache.py \
  --epochs data/processed/StudyD \
  --manifest splits/heldout_manifest.json \
  --source raw \
  --raw-dir data/raw/bigP3BCI_dataset/bigP3BCI-data \
  --l-freq 1 --h-freq 12 --no-baseline-correction \
  --output data/cache/classifier_1_12_no_baseline.h5

# Reproduce the locked manifest evaluation (requires explicit held-out evaluation opt-in).
python scripts/evaluate_classifier_manifest.py \
  --allow-heldout-eval \
  --cache data/cache/classifier_1_12_no_baseline.h5 \
  --config configs/classifier_optimization/preprocessing/baseline_off.json \
  --lock splits/finalist_lock.json \
  --manifest splits/heldout_manifest.json \
  --output results/tables/classifier_manifest_eval.json

# Optional: rebuild selected cache directly from raw EDF with explicit filter values.
# This still consumes existing FIF names/metadata to choose training run IDs.
python scripts/build_classifier_cache.py \
  --epochs data/processed/StudyD \
  --manifest splits/heldout_manifest.json \
  --source raw \
  --raw-dir data/raw/bigP3BCI_dataset/bigP3BCI-data-backup \
  --l-freq 0.1 --h-freq 30.0 \
  --output data/cache/classifier_epochs_raw.h5

# Study Q clean training cache (its builder uses its configured Study Q input root).
python scripts/build_study_q_classifier_cache.py \
  --output data/cache/study_q_classifier_epochs.h5

# P7/P8 paired cache builds: same clean Study D runs; vary one preprocessing factor.
python scripts/build_classifier_cache.py --epochs data/processed/StudyD --manifest splits/heldout_manifest.json --source raw --raw-dir data/raw/bigP3BCI_dataset/bigP3BCI-data --l-freq 0.1 --h-freq 30 --output data/cache/preproc_ablation_filter0130_base_on.h5
python scripts/build_classifier_cache.py --epochs data/processed/StudyD --manifest splits/heldout_manifest.json --source raw --raw-dir data/raw/bigP3BCI_dataset/bigP3BCI-data --no-bandpass --output data/cache/preproc_ablation_unfiltered_base_on.h5
python scripts/build_classifier_cache.py --epochs data/processed/StudyD --manifest splits/heldout_manifest.json --source raw --raw-dir data/raw/bigP3BCI_dataset/bigP3BCI-data --l-freq 0.1 --h-freq 30 --no-baseline-correction --output data/cache/preproc_ablation_filter0130_base_off.h5

# Run only tuned SWLDA on each cache (M1 uses identical run-grouped folds/settings).
python scripts/run_classical_cv.py --cache data/cache/preproc_ablation_filter0130_base_on.h5 --config configs/classifier_optimization/preprocessing/baseline_on.json --manifest splits/heldout_manifest.json --folds 3 --max-fit-samples 3000 --models M1_tuned_SWLDA --output results/tables/preproc_ablation_filter_on.json --scores-output results/tables/preprocessing_ablation_filter0130_baseline_on_scores.npz
python scripts/run_classical_cv.py --cache data/cache/preproc_ablation_unfiltered_base_on.h5 --config configs/classifier_optimization/preprocessing/baseline_on.json --manifest splits/heldout_manifest.json --folds 3 --max-fit-samples 3000 --models M1_tuned_SWLDA --output results/tables/preproc_ablation_unfiltered_on.json --scores-output results/tables/preprocessing_ablation_unfiltered_baseline_on_scores.npz
python scripts/run_classical_cv.py --cache data/cache/preproc_ablation_filter0130_base_off.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --folds 3 --max-fit-samples 3000 --models M1_tuned_SWLDA --output results/tables/preproc_ablation_filter_off.json --scores-output results/tables/preprocessing_ablation_filter0130_baseline_off_scores.npz
python scripts/run_classical_cv.py --cache data/cache/classifier_1_12_no_baseline.h5 --config configs/classifier_optimization/preprocessing/baseline_off.json --manifest splits/heldout_manifest.json --folds 3 --max-fit-samples 3000 --models M1_tuned_SWLDA --output results/tables/preproc_ablation_filter1_12_baseline_off.json --scores-output results/tables/preproc_ablation_filter1_12_baseline_off_scores.npz
python scripts/bootstrap_preprocessing_ablation.py --reference results/tables/preprocessing_ablation_unfiltered_baseline_on_scores.npz --candidate results/tables/preprocessing_ablation_filter0130_baseline_on_scores.npz --reference-label "no bandpass, baseline on" --candidate-label "0.1-30 Hz, baseline on" --output results/tables/preproc_ablation_filter_bootstrap_p7.json --n-bootstrap 10000 --seed 0
python scripts/bootstrap_preprocessing_ablation.py --reference results/tables/preprocessing_ablation_filter0130_baseline_off_scores.npz --candidate results/tables/preprocessing_ablation_filter0130_baseline_on_scores.npz --reference-label "0.1-30 Hz, baseline off" --candidate-label "0.1-30 Hz, baseline on" --output results/tables/preproc_ablation_filter_bootstrap_p8.json --n-bootstrap 10000 --seed 0
python scripts/bootstrap_preprocessing_ablation.py --reference results/tables/preprocessing_ablation_filter0130_baseline_off_scores.npz --candidate results/tables/preproc_ablation_filter1_12_baseline_off_scores.npz --reference-label "0.1-30 Hz, baseline off" --candidate-label "1-12 Hz, baseline off" --output results/tables/preproc_ablation_filter_bootstrap.json --n-bootstrap 10000 --seed 0
```

The optional RP/CWT regeneration command is archived on the `CNN` dead-arm branch, not in the current mainline:

```bash
python scripts/compute_2d_features.py
```

## Classifier handoff (10 lines)

1. Parser output: MNE FIF epochs `X=(epochs, EEG channels, samples)`, binary event label `0=non-target`, `1=target`.
2. Per-epoch metadata: `stimulus_code`, `stimulus_type`; adaptive adds 0-based `char_index` and 1-based `sequence_in_char`; group flash adds row-major `lit_mask`.
3. HDF5 is a separate classifier-cache interface; cache schema is deferred to the classifier component.
4. Batch parser/FIF defaults: 0.1–30 Hz, no notch, −0.1…0.8 s, baseline on, no rejection, decim 1.
5. Separate clean-reference cache `classifier_epochs.h5`: 83 RC/Train runs, 84,660 epochs, 0.1–30 Hz, baseline on; SHA is in `splits/baseline_d_lock.json`.
6. Final Study D manifest classifier evaluation (baseline M0 and finalist M3a): effective 1–12 Hz, baseline off, classifier window 0…0.8 s, decim 2; direct raw-EDF parse is pinned by `splits/finalist_lock.json` and result JSON.
7. Study Q cache: 1–12 Hz, −0.1…0.8 s, baseline off, decim 2, 12-flash group sequence.
8. Study D training labels: 9×8 grid; 9,960 targets vs 74,700 non-targets (1:7.5).
9. Baseline-off M1 comparison: 1–12 vs 0.1–30 Hz AUC 0.7599 vs 0.7350; character accuracy 67.07% vs 62.05% (498 characters).
10. Paired run-bootstrap 95% CI: AUC +1.94 to +3.06 pp; accuracy +0.80 to +9.24 pp; P9 is M1 evidence, not paper-final M0.
