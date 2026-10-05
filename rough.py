import sys, numpy as np, mne
for p in sys.argv[1:]:
    raw = mne.io.read_raw_edf(p, preload=True, verbose=False)
    d = raw.get_data()
    print(f"\n== {p}")
    print("sfreq:", raw.info["sfreq"], "| n_channels:", len(raw.ch_names),
          "| duration_s:", round(raw.times[-1], 1))
    print("channels:", raw.ch_names)
    for i, n in enumerate(raw.ch_names):
        if any(k in n for k in ("Stimulus", "Target", "Phase", "Run", "Trial")):
            v = np.unique(d[i])
            print(f"  {n}: {len(v)} unique", v[:15])