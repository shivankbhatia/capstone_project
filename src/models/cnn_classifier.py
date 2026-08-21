"""CNN P300 classifier over RP + CWT 2D features (new ablation arm).

Does NOT replace src/models/classifier.py (SGD/SWLDA on raw epochs) -- that
stays as-is for the existing Baseline/Fixed/Adaptive/RAG arms. This is an
additional model, trained on the outputs of scripts/compute_2d_features.py,
meant to be compared as a 6th ablation condition ("CNN (RP+CWT)") in
run_pipeline.py once trained.

Input per epoch: recurrence_2d (8, 2080) reshaped to (8, 64, 64) via
reconstruct_recurrence_plot, concatenated channel-wise with cwt_2d (8, 30, 64)
resized to (8, 64, 64) -- two 8-channel "image stacks" fed to parallel conv
branches, fused before the classification head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RPBranch(nn.Module):
    """Conv branch over (B, 8, 64, 64) recurrence-plot images."""

    def __init__(self, in_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> (B,16,32,32)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> (B,32,16,16)
        return x


class CWTBranch(nn.Module):
    """Conv branch over (B, 8, 30, 64) CWT spectrogram images."""

    def __init__(self, in_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 5), padding=(1, 2))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool2d((16, 16))
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)                                  # -> (B,32,16,16)
        return x


class P300CNN(nn.Module):
    """Fuses RP + CWT branches, outputs P(target) logit."""

    def __init__(self, n_channels=8):
        super().__init__()
        self.rp_branch = RPBranch(n_channels)
        self.cwt_branch = CWTBranch(n_channels)
        fused_dim = (32 * 16 * 16) * 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, rp_img, cwt_img):
        rp_feat = self.rp_branch(rp_img)
        cwt_feat = self.cwt_branch(cwt_img)
        fused = torch.cat([rp_feat.flatten(1), cwt_feat.flatten(1)], dim=1)
        return self.head(fused).squeeze(-1)


def prepare_batch(rp_triu, cwt, n_times=64):
    """Convert stored (B, C, triu_len) + (B, C, F, T) into CNN-ready tensors.

    rp_triu : (B, C, T*(T+1)/2) as stored in recurrence_2d
    cwt     : (B, C, F, T) as stored in cwt_2d, resized to (B, C, 64, 64)

    Returns (rp_img, cwt_img), both (B, C, 64, 64) float32 torch tensors.

    Fully vectorized: triu indices are computed once (module-level cache)
    and scatter-assigned across the whole (B, C) batch at once, instead of
    looping reconstruct_recurrence_plot per sample per channel (which
    recomputed np.triu_indices(64) on every single call -- the dominant
    cost in the original implementation).
    """
    import numpy as np

    b, c, triu_len = rp_triu.shape
    r_idx, c_idx = _get_triu_idx(n_times)

    rp_img = np.zeros((b, c, n_times, n_times), dtype=np.float32)
    rp_img[:, :, r_idx, c_idx] = rp_triu.astype(np.float32)
    rp_img[:, :, c_idx, r_idx] = rp_img[:, :, r_idx, c_idx]

    rp_img_t = torch.from_numpy(rp_img)
    cwt_t = torch.from_numpy(cwt.astype(np.float32))
    cwt_img_t = F.interpolate(cwt_t, size=(n_times, n_times), mode="bilinear",
                               align_corners=False)
    return rp_img_t, cwt_img_t


_TRIU_CACHE = {}


def _get_triu_idx(n_times):
    if n_times not in _TRIU_CACHE:
        import numpy as np
        _TRIU_CACHE[n_times] = np.triu_indices(n_times)
    return _TRIU_CACHE[n_times]