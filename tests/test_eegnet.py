import numpy as np
import torch

from src.models.eegnet import EEGNet, sequence_competition_loss


def test_eegnet_outputs_one_flash_logit():
    model = EEGNet(n_channels=4, n_times=96)
    output = model(torch.randn(3, 4, 96))
    assert output.shape == (3,)


def test_sequence_competition_loss_is_finite_for_target_row_and_column():
    codes = torch.arange(1, 18).repeat(2, 1)
    labels = torch.zeros_like(codes, dtype=torch.float32)
    labels[:, 2] = 1.0
    labels[:, 10] = 1.0
    logits = torch.where(labels > 0, 2.0, -1.0).requires_grad_(True)
    loss = sequence_competition_loss(logits, labels, codes, mode="both")
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
