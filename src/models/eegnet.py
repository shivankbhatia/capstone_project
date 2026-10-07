"""Small EEGNet-style per-flash P300 classifier."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.models.classifier_interface import P300Classifier


def sequence_competition_loss(
    logits: torch.Tensor, labels: torch.Tensor, stimulus_codes: torch.Tensor,
    *, mode: str = "both", positive_weight: float = 7.0, sequence_weight: float = 1.0,
) -> torch.Tensor:
    """Combine weighted flash BCE with row-vs-row and column-vs-column CE.

    Inputs are complete 17-flash row/column sequences, shaped ``(batch, 17)``.
    Row codes use 1..9 and column codes use 10..17.
    """
    if mode not in {"bce", "softmax", "both"}:
        raise ValueError("mode must be 'bce', 'softmax', or 'both'")
    if logits.shape != labels.shape or logits.shape != stimulus_codes.shape:
        raise ValueError("logits, labels, and stimulus_codes must have equal shapes")
    row_scores = torch.zeros((len(logits), 9), dtype=logits.dtype, device=logits.device)
    col_scores = torch.zeros((len(logits), 8), dtype=logits.dtype, device=logits.device)
    row_mask = (stimulus_codes >= 1) & (stimulus_codes <= 9)
    col_mask = (stimulus_codes >= 10) & (stimulus_codes <= 17)
    row_scores.scatter_add_(1, (stimulus_codes - 1).clamp(0, 8), logits * row_mask)
    col_scores.scatter_add_(1, (stimulus_codes - 10).clamp(0, 7), logits * col_mask)

    target_rows = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    target_cols = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    for i in range(len(logits)):
        positive_codes = stimulus_codes[i][labels[i] > 0.5]
        rows = positive_codes[(positive_codes >= 1) & (positive_codes <= 9)] - 1
        cols = positive_codes[(positive_codes >= 10) & (positive_codes <= 17)] - 10
        if rows.numel() == 0 or cols.numel() == 0:
            raise ValueError("Each sequence must include one target row and one target column")
        target_rows[i] = torch.mode(rows).values
        target_cols[i] = torch.mode(cols).values

    sequence_loss = F.cross_entropy(row_scores, target_rows) + F.cross_entropy(col_scores, target_cols)
    if mode == "softmax":
        return sequence_loss
    bce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), pos_weight=torch.tensor(positive_weight, device=logits.device)
    )
    if mode == "bce":
        return bce
    return bce + sequence_weight * sequence_loss


class EEGNet(nn.Module):
    def __init__(self, n_channels: int, n_times: int, dropout: float = 0.35):
        super().__init__()
        temporal_kernel = min(64, max(16, (n_times // 2) | 1))
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(16),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(n_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding=(0, 8), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("EEGNet expects (batch, channels, time)")
        x = x.unsqueeze(1)
        return self.separable(self.spatial(self.temporal(x))).squeeze(-1)


class EEGNetClassifier(P300Classifier):
    """Trainable EEGNet exposing calibrated flash logits through P300Classifier."""

    def __init__(
        self, seed: int = 42, epochs: int = 30, patience: int = 6,
        learning_rate: float = 1e-3, weight_decay: float = 1e-4,
        class_weight: float = 7.0, batch_size: int = 256,
    ):
        super().__init__()
        self.seed = int(seed)
        self.max_epochs = int(epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.class_weight = float(class_weight)
        self.batch_size = int(batch_size)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.ndim != 3 or len(X) != len(y):
            raise ValueError("X must have shape (epochs, channels, time) and align with y")
        if groups is None:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            train_idx, val_idx = next(splitter.split(X, y))
        else:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
            train_idx, val_idx = next(splitter.split(X, y, groups=np.asarray(groups)))

        model = EEGNet(X.shape[1], X.shape[2]).to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.class_weight], device=self.device)
        )
        best_loss = float("inf")
        best_state = None
        wait = 0
        swa_model = AveragedModel(model)
        swa_count = 0

        x_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        x_val = torch.from_numpy(X[val_idx])
        y_val = torch.from_numpy(y[val_idx])
        scale = float(np.std(X[train_idx]))

        for epoch in range(self.max_epochs):
            model.train()
            order = torch.randperm(len(x_train))
            for start in range(0, len(order), self.batch_size):
                batch_idx = order[start:start + self.batch_size]
                xb = x_train[batch_idx].to(self.device)
                yb = y_train[batch_idx].to(self.device)
                # Mild time shift, sensor noise, and channel dropout.
                shift = int(np.random.randint(-4, 5))
                if shift:
                    xb = torch.roll(xb, shifts=shift, dims=-1)
                xb = xb + torch.randn_like(xb) * (0.10 * scale)
                channel_mask = (torch.rand((len(xb), xb.shape[1], 1), device=self.device) > 0.05)
                xb = xb * channel_mask
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xb).reshape(-1), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for start in range(0, len(x_val), self.batch_size):
                    xb = x_val[start:start + self.batch_size].to(self.device)
                    yb = y_val[start:start + self.batch_size].to(self.device)
                    total_loss += float(criterion(model(xb).reshape(-1), yb).item()) * len(xb)
                val_loss = total_loss / len(x_val)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if epoch >= self.max_epochs - 5:
                swa_model.update_parameters(model)
                swa_count += 1
            if wait >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        if swa_count >= 3:
            # Average the best checkpoint and final-epoch SWA weights.
            avg = swa_model.module.state_dict()
            for key, value in model.state_dict().items():
                if value.is_floating_point() and key in avg:
                    value.copy_(0.5 * value + 0.5 * avg[key].to(value.device))
        model.eval()
        self.model = model
        return self

    def fit_sequence(
        self, X: np.ndarray, y: np.ndarray, stimulus_codes: np.ndarray,
        groups, *, loss_mode: str = "both", sequence_weight: float = 1.0,
    ):
        """Fit from complete 17-flash row/column sequences.

        ``X`` is ``(n_sequences, 17, channels, time)``. ``groups`` contains
        one run ID per sequence and is used for an inner run-grouped validation
        split for early stopping.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.uint8)
        stimulus_codes = np.asarray(stimulus_codes, dtype=np.int64)
        groups = np.asarray(groups)
        if X.ndim != 4 or X.shape[1] != 17 or y.shape != X.shape[:2]:
            raise ValueError("Expected X (sequences, 17, channels, time) and aligned labels")
        if stimulus_codes.shape != y.shape or len(groups) != len(y):
            raise ValueError("Stimulus codes and run groups must align with sequences")
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=self.seed)
        train_idx, val_idx = next(splitter.split(X, y[:, 0], groups=groups))
        model = EEGNet(X.shape[2], X.shape[3]).to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        best_loss, best_state, wait = float("inf"), None, 0
        swa_model, swa_count = AveragedModel(model), 0
        x_train = torch.from_numpy(X[train_idx])
        y_train = torch.from_numpy(y[train_idx])
        c_train = torch.from_numpy(stimulus_codes[train_idx])
        x_val = torch.from_numpy(X[val_idx])
        y_val = torch.from_numpy(y[val_idx])
        c_val = torch.from_numpy(stimulus_codes[val_idx])
        scale = float(np.std(X[train_idx]))

        for epoch in range(self.max_epochs):
            model.train()
            order = torch.randperm(len(x_train))
            for start in range(0, len(order), max(1, self.batch_size // 17)):
                batch_idx = order[start:start + max(1, self.batch_size // 17)]
                xb = x_train[batch_idx].to(self.device)
                yb = y_train[batch_idx].to(self.device).float()
                cb = c_train[batch_idx].to(self.device)
                shift = int(np.random.randint(-4, 5))
                if shift:
                    xb = torch.roll(xb, shifts=shift, dims=-1)
                xb = xb + torch.randn_like(xb) * (0.10 * scale)
                channel_mask = torch.rand(
                    (len(xb), 1, xb.shape[2], 1), device=self.device
                ) > 0.05
                xb = xb * channel_mask
                optimizer.zero_grad(set_to_none=True)
                batch_logits = model(xb.flatten(0, 1)).reshape(len(xb), 17)
                loss = sequence_competition_loss(
                    batch_logits, yb, cb, mode=loss_mode,
                    positive_weight=self.class_weight, sequence_weight=sequence_weight,
                )
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                total_loss = 0.0
                for start in range(0, len(x_val), max(1, self.batch_size // 17)):
                    stop = min(len(x_val), start + max(1, self.batch_size // 17))
                    xb = x_val[start:stop].to(self.device)
                    yb = y_val[start:stop].to(self.device).float()
                    cb = c_val[start:stop].to(self.device)
                    val_logits = model(xb.flatten(0, 1)).reshape(len(xb), 17)
                    loss = sequence_competition_loss(
                        val_logits, yb, cb, mode=loss_mode,
                        positive_weight=self.class_weight, sequence_weight=sequence_weight,
                    )
                    total_loss += float(loss.item()) * len(xb)
                val_loss = total_loss / len(x_val)
            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if epoch >= self.max_epochs - 5:
                swa_model.update_parameters(model)
                swa_count += 1
            if wait >= self.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        if swa_count >= 3:
            avg = swa_model.module.state_dict()
            for key, value in model.state_dict().items():
                if value.is_floating_point() and key in avg:
                    value.copy_(0.5 * value + 0.5 * avg[key].to(value.device))
        model.eval()
        self.model = model
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit before decision_function")
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = X[start:start + self.batch_size].to(self.device)
                logits.append(self.model(batch).reshape(-1).detach().cpu().numpy())
        return np.concatenate(logits) if logits else np.empty(0, dtype=np.float64)
