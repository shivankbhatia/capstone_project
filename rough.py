import mne, numpy as np
raw = mne.io.read_raw_edf("data/raw/bigP3BCI_dataset/bigP3BCI-data/StudyD/D_04/SE001/Test/Dyn/D_04_SE001_Dyn_Test01.edf", preload=True, verbose=False)
selected_target = np.round(raw.get_data(picks=['SelectedTarget'])[0]).astype(int)

rising_edges = np.where((selected_target[:-1] == 0) & (selected_target[1:] != 0))[0] + 1
print("Pulse codes (at edge):", selected_target[rising_edges].tolist())

# Look a few samples AFTER each edge, to see if the value is still settling
for i, edge in enumerate(rising_edges):
    window = selected_target[edge : edge + 10]
    print(f"  Pulse {i}: edge_idx={edge}, next 10 samples: {window.tolist()}")