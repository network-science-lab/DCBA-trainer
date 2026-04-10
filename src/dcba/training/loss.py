"""Loss functions — Phase 2+ stubs."""

import torch.nn as nn


class IdeaALoss(nn.Module):
    """Weighted sum of embedding discrepancy and autoencoder reconstruction loss."""

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError
