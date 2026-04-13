"""Lightning wrapper for the config autoencoder — Phase 1 stub."""

import lightning.pytorch as pl


class DCBAWrapper(pl.LightningModule):
    """LightningModule wrapping the config autoencoder for training and evaluation."""

    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError
