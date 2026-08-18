import io
import os
import contextlib
from pathlib import Path

import numpy as np
import mne

from src.data.batch_preprocess import parse_bigp3bci_edf
import run_pipeline

project = Path('/Users/shivank/Projects/P300_EEG_Speller/p300-llm-speller')
raw_root = project / 'data' / 'raw' / 'bigP3BCI_dataset' / 'bigP3BCI-data' / 'StudyD'
processed_root = project / 'data' / 'processed' / 'StudyD'

button_raw = raw_root / 'D_02' / 'SE001' / 'Test' / 'Dyn' / 'D_02_SE001_Dyn_Test02.edf'
rc_raw = raw_root / 'D_01' / 'SE001' / 'Train' / 'RC' / 'D_01_SE001_RC_Train01.edf'

print('=== Adaptive Spot Check: BUTTON file ===')
print(f'Raw file: {button_raw}')
epochs, spelled, grid_info = parse_bigp3bci_edf(str(button_raw), verbose=False)
print(f'Decoded spelled string: {spelled}')
print(f'Metadata columns: {list(epochs.metadata.columns)}')

md = epochs.metadata.copy()
valid = md[md['char_index'] >= 0]
distinct_chars = sorted(valid['char_index'].unique().tolist())
print(f'Distinct char_index values (non-negative): {distinct_chars}')
print(f'Count distinct char_index: {len(distinct_chars)}')

n_rows = grid_info['n_rows']
n_cols = grid_info['n_cols']
flashes_per_seq = n_rows + n_cols
print(f'Grid rows/cols: {n_rows}/{n_cols}, flashes_per_seq: {flashes_per_seq}')

for cidx in distinct_chars:
    rows = valid[valid['char_index'] == cidx]
    seq_vals = sorted(rows['sequence_in_char'].unique().tolist())
    expected = list(range(1, max(seq_vals) + 1)) if seq_vals else []
    is_consecutive = seq_vals == expected
    first_val = seq_vals[0] if seq_vals else None
    print(
        f'char_index={cidx}: unique sequence_in_char={seq_vals}, '
        f'resets_to_1={first_val == 1}, consecutive={is_consecutive}, '
        f'flash_count={len(rows)}'
    )

raw = mne.io.read_raw_edf(str(button_raw), preload=False, verbose=False)
all_ch = raw.ch_names
stim_begin_ch = next(ch for ch in all_ch if ch.strip().lower() == 'stimulusbegin')
selected_target_ch = next(ch for ch in all_ch if ch.strip().lower() == 'selectedtarget')

stim_begin = raw.get_data(picks=[stim_begin_ch])[0]
threshold = 0.5
rising_edges = np.where((stim_begin[:-1] <= threshold) & (stim_begin[1:] > threshold))[0] + 1

selected_target = np.round(raw.get_data(picks=[selected_target_ch])[0]).astype(int)
pulse_edges = np.where((selected_target[:-1] == 0) & (selected_target[1:] > 0))[0] + 1
pulse_codes = selected_target[pulse_edges]

event_samples_kept = rising_edges[epochs.selection]
char_idx_arr = md['char_index'].to_numpy()

char2_samples = event_samples_kept[char_idx_arr == 2]
char3_samples = event_samples_kept[char_idx_arr == 3]

boundary_sample = int(pulse_edges[2])
boundary_code = int(pulse_codes[2])
next_boundary_sample = int(pulse_edges[3])
next_boundary_code = int(pulse_codes[3])

char2_last = int(char2_samples.max()) if len(char2_samples) else None
char3_first = int(char3_samples.min()) if len(char3_samples) else None

print('Doubled-letter boundary evidence (T/T):')
print(f'  pulse_edges[2] sample={boundary_sample}, code={boundary_code}')
print(f'  pulse_edges[3] sample={next_boundary_sample}, code={next_boundary_code}')
print(f'  last sample with char_index==2: {char2_last}')
print(f'  first sample with char_index==3: {char3_first}')
print(
    f'  boundary ordering ok: '
    f'{char2_last is not None and char3_first is not None and char2_last < boundary_sample < char3_first < next_boundary_sample}'
)

button_out = processed_root / f'{button_raw.stem}-epo.fif'
epochs.save(str(button_out), overwrite=True)
print(f'Regenerated adaptive epochs with metadata: {button_out}')

print('\n=== RC/Train Spot Check ===')
print(f'Raw file: {rc_raw}')
rc_epochs, rc_spelled, _ = parse_bigp3bci_edf(str(rc_raw), verbose=False)
print(f'Decoded spelled string (RC file): {rc_spelled}')
print(f'Metadata columns: {list(rc_epochs.metadata.columns)}')
print(f"Has char_index column: {'char_index' in rc_epochs.metadata.columns}")
print(f"Has sequence_in_char column: {'sequence_in_char' in rc_epochs.metadata.columns}")

rc_out = processed_root / f'{rc_raw.stem}-epo.fif'
rc_epochs.save(str(rc_out), overwrite=True)
print(f'Regenerated RC epochs: {rc_out}')

print('\n=== Fallback Warning Visibility Check ===')
legacy_out = processed_root / 'TMP_LEGACY_BUTTON_NO_ADAPTIVE_META-epo.fif'
legacy_epochs = epochs.copy()
legacy_epochs.metadata = legacy_epochs.metadata.drop(columns=['char_index', 'sequence_in_char'])
legacy_epochs.save(str(legacy_out), overwrite=True)

class DummyClf:
    def predict_proba(self, X):
        n = X.shape[0]
        probs = np.zeros((n, 2), dtype=float)
        probs[:, 0] = 0.4
        probs[:, 1] = 0.6
        return probs

stdout_buf = io.StringIO()
with contextlib.redirect_stdout(stdout_buf):
    _ = run_pipeline.real_eeg_classifier_stream(
        str(legacy_out),
        char_idx=0,
        sequence_num=1,
        clf=DummyClf(),
        char_list=[''] * (n_rows * n_cols),
        n_rows=n_rows,
        n_cols=n_cols,
        flashes_per_seq=flashes_per_seq,
        max_seqs_per_char=15,
    )
warning_output = stdout_buf.getvalue().strip()
print('Captured warning output:')
print(warning_output if warning_output else '[NO WARNING PRINTED]')

try:
    os.remove(legacy_out)
except OSError:
    pass

print('\n=== Summary Checks ===')
print(f"Adaptive distinct char_index == [0,1,2,3,4,5]: {distinct_chars == [0, 1, 2, 3, 4, 5]}")
print(f"Adaptive decoded word == BUTTON: {spelled == 'BUTTON'}")
print(
    f"RC has no adaptive columns (expected current behavior): "
    f"{('char_index' not in rc_epochs.metadata.columns and 'sequence_in_char' not in rc_epochs.metadata.columns)}"
)
print(f"Fallback warning visible: {'legacy fixed-block flash indexing' in warning_output}")
