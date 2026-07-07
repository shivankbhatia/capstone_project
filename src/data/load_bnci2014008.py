"""
Day 1 — Setup & scoping
Loads BNCI2014-008 (P300 speller, 8 ALS subjects) via MOABB and inspects
its structure so we know exactly what we're working with before
preprocessing starts on Day 2.

Run: python -m src.data.load_bnci2014008
"""

from moabb.datasets import BNCI2014_008
from moabb.paradigms import P300

import mne


def main():
    dataset = BNCI2014_008()
    print(f"Dataset code: {dataset.code}")
    print(f"Subjects: {dataset.subject_list}")

    subject_id = dataset.subject_list[0]
    sessions = dataset.get_data(subjects=[subject_id])

    subject_data = sessions[subject_id]
    print(f"\nSubject {subject_id} sessions: {list(subject_data.keys())}")

    for session_name, runs in subject_data.items():
        print(f"\nSession: {session_name}")
        for run_name, raw in runs.items():
            print(f"  Run: {run_name}")
            print(f"    Channels: {raw.ch_names}")
            print(f"    Sampling rate: {raw.info['sfreq']} Hz")
            print(f"    Duration: {raw.n_times / raw.info['sfreq']:.1f} s")

            events, event_id = mne.events_from_annotations(raw)
            print(f"    Event types: {event_id}")
            print(f"    Number of events: {len(events)}")
            break
        break

    paradigm = P300(resample=128)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])

    print(f"\nEpoched data shape (trials, channels, samples): {X.shape}")
    import numpy as np
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    print(f"Metadata columns: {list(meta.columns)}")


if __name__ == "__main__":
    main()
