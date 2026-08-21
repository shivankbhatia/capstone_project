"""2D feature transforms for P300 epochs: Recurrence Plots and CWT spectrograms.

Adapted from the bigP3BCI RP/CWT preprocessing techniques, but generalized to
this project's actual channel set, native sampling rate, and epoch window
(-100 to 800 ms) instead of the fixed 8-channel / 128 Hz assumption. Existing
preprocessing (epoching.py / batch_preprocess.py), the classifier, fusion,
LLM, and RAG layers are untouched -- these are additive feature extractors
only, computed on top of already-epoched data.

Space optimization: RP/CWT are computed on a canonical 8-channel P300 subset
and a downsampled time axis (RP_CWT_N_TIMES), independent of the classifier's
native 256 Hz / full-montage data used everywhere else in the pipeline.
"""
import gc

import mne
import numpy as np

try:
    import torch
    _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    torch = None
    _DEVICE = None

# Canonical P300 channel subset for RP/CWT only. Falls back to all available
# channels (case-insensitive match) if fewer than this are present.
RP_CWT_CHANNELS = ['Fz', 'Cz', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

# Time axis is decimated to this many samples before RP/CWT (RP cost is
# O(T^2), so this is the main lever on storage). Does not touch the
# classifier's native-rate data anywhere else.
RP_CWT_N_TIMES = 64


def select_rp_cwt_channels(ch_names):
    """Return indices of the canonical 8-channel subset within ch_names.

    Case-insensitive, tolerant of 'EEG_' prefixes already stripped upstream.
    Falls back to the first 8 channels (with a warning) if fewer than 8 of
    the canonical names are found, so this never hard-fails on montages
    that don't include all of them.
    """
    lower_map = {ch.lower(): i for i, ch in enumerate(ch_names)}
    idx = [lower_map[c.lower()] for c in RP_CWT_CHANNELS if c.lower() in lower_map]

    if len(idx) < len(RP_CWT_CHANNELS):
        missing = [c for c in RP_CWT_CHANNELS if c.lower() not in lower_map]
        print(f"WARNING: RP/CWT channel subset missing {missing} in this "
              f"montage; falling back to first {min(8, len(ch_names))} channels.")
        idx = list(range(min(8, len(ch_names))))

    return idx


def _decimate_time(epochs_data, n_times_out):
    """Decimate the time axis to n_times_out samples via uniform index pick.

    Cheap and deterministic; avoids importing scipy.signal.resample just for
    this. Fine for RP/CWT (exploratory 2D features) -- the classifier's
    native-rate data is untouched.
    """
    n_times_in = epochs_data.shape[-1]
    if n_times_in <= n_times_out:
        return epochs_data
    idx = np.linspace(0, n_times_in - 1, n_times_out).round().astype(int)
    return epochs_data[..., idx]


def _triu_size(n_times):
    return n_times * (n_times + 1) // 2


def compute_recurrence_plots(epochs_data, batch_size=100):
    """Gaussian-kernelized recurrence plots, upper-triangular-vectorized.

    R_ij = exp(-|x_i - x_j|^2 / (2 * sigma^2)), sigma = std(x) per
    epoch/channel (adaptive bandwidth, as in the reference implementation).

    Parameters
    ----------
    epochs_data : ndarray, shape (N, C, T) -- already channel-subset and
        time-decimated by the caller (see compute_2d_features.py).
    batch_size : int

    Returns
    -------
    ndarray, shape (N, C, T*(T+1)/2), float16
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    triu_len = _triu_size(n_times)
    rp_tensors = np.empty((n_epochs, n_channels, triu_len), dtype=np.float16)

    use_gpu = torch is not None and _DEVICE is not None and _DEVICE.type == 'cuda'

    if use_gpu:
        triu_r, triu_c = torch.triu_indices(n_times, n_times, device=_DEVICE)
        for i in range(0, n_epochs, batch_size):
            end = min(i + batch_size, n_epochs)
            batch_t = torch.tensor(epochs_data[i:end], dtype=torch.float32, device=_DEVICE)

            std_t = torch.std(batch_t, dim=-1, keepdim=True)
            std_t = torch.clamp(std_t, min=1e-9).unsqueeze(-1)

            dist_t = torch.abs(batch_t.unsqueeze(-1) - batch_t.unsqueeze(-2))
            rp_t = torch.exp(-(dist_t ** 2) / (2 * (std_t ** 2)))

            rp_tensors[i:end] = rp_t[:, :, triu_r, triu_c].cpu().numpy().astype(np.float16)
            del batch_t, std_t, dist_t, rp_t
            torch.cuda.empty_cache()
    else:
        triu_r, triu_c = np.triu_indices(n_times)
        for i in range(0, n_epochs, batch_size):
            end = min(i + batch_size, n_epochs)
            batch = epochs_data[i:end]

            std = np.clip(np.std(batch, axis=-1, keepdims=True), 1e-9, None)[..., None]
            dist = np.abs(batch[:, :, :, None] - batch[:, :, None, :])
            rp_full = np.exp(-(dist ** 2) / (2 * (std ** 2)))

            rp_tensors[i:end] = rp_full[:, :, triu_r, triu_c].astype(np.float16)
            del std, dist, rp_full
            gc.collect()

    return rp_tensors


def reconstruct_recurrence_plot(triu_vector, n_times, triu_idx=None):
    """Rebuild the symmetric (T, T) matrix from an upper-triangular vector.

    Pass a cached triu_idx = np.triu_indices(n_times) to avoid recomputing
    it on every call -- this is the dominant cost in batch reconstruction
    (see prepare_batch_vectorized in cnn_classifier.py).
    """
    rp = np.zeros((n_times, n_times), dtype=triu_vector.dtype)
    r_idx, c_idx = triu_idx if triu_idx is not None else np.triu_indices(n_times)
    rp[r_idx, c_idx] = triu_vector
    rp[c_idx, r_idx] = rp[r_idx, c_idx]
    return rp


def compute_cwt_spectrograms(epochs_data, sfreq, batch_size=100, n_freqs=30,
                              f_min=1.0, f_max=30.0):
    """Morlet-wavelet CWT magnitude spectrograms.

    30 log-spaced frequencies in [f_min, f_max] Hz, dynamic cycles
    clip(freqs / 6.0, 0.5, 5.0) for temporal localization of the P300.

    Parameters
    ----------
    epochs_data : ndarray, shape (N, C, T) -- already channel-subset and
        time-decimated by the caller (see compute_2d_features.py).
    sfreq : float -- the EFFECTIVE rate after time decimation (pass the
        decimated sfreq, not the classifier's native 256 Hz), so frequency
        bins stay physically meaningful.

    Returns
    -------
    ndarray, shape (N, C, F, T), float16
    """
    n_epochs, n_channels, n_times = epochs_data.shape
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), num=n_freqs)
    n_cycles = np.clip(freqs / 6.0, 0.5, 5.0)
    cwt_tensors = np.empty((n_epochs, n_channels, n_freqs, n_times), dtype=np.float16)

    for i in range(0, n_epochs, batch_size):
        end = min(i + batch_size, n_epochs)
        batch = epochs_data[i:end]
        cwt_complex = mne.time_frequency.tfr_array_morlet(
            batch, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
            output='complex', verbose=False,
        )
        cwt_tensors[i:end] = np.abs(cwt_complex).astype(np.float16)
        del cwt_complex
        gc.collect()

    return cwt_tensors