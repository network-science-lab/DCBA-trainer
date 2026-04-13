"""Config encoder module — Phase 1 stub."""

import torch.nn as nn


class ConfigEncoder(nn.Module):
    """MLP autoencoder that maps a config vector ``q`` to an embedding ``h_q`` and back."""

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError
