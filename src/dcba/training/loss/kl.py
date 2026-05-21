"""KL divergence loss for VAE regularisation."""

import torch
import torch.nn as nn
from torch import Tensor


class KLDivergenceLoss(nn.Module):
    """Closed-form KL divergence of N(mu, sigma^2) against a standard normal N(0, 1)."""

    def forward(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """
        Compute scalar KL divergence averaged over the batch.

        Uses the closed form: mean over batch of ``-0.5 * sum(1 + 2*log_sigma - mu^2 - sigma^2)``.

        :param mu: Mean tensor of shape ``(batch, embedding_dim)``.
        :param log_sigma: Log standard deviation tensor of shape ``(batch, embedding_dim)``.

        :returns: Scalar KL divergence.
        """
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * log_sigma - mu.pow(2) - torch.exp(2 * log_sigma), dim=-1)
        )
