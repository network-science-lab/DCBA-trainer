"""Lightning wrappers for DCBA training regimes.

Each training regime (Phase 1 autoencoder, Phase 2 joint, …) has its own
:class:`~lightning.pytorch.LightningModule` subclass.  The :func:`~dcba.training.trainer.train`
function selects the appropriate wrapper from config, keeping each wrapper focused and
free of mode-switch flags.
"""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import Tensor

from dcba.models.config_encoder import ConfigEncoder


class ConfigAutoencoderWrapper(pl.LightningModule):
    """
    LightningModule for Phase 1: trains the config autoencoder in isolation.

    Computes MSE reconstruction loss between the input config vector and the decoder
    output.  Logs ``{stage}_loss`` at every step.

    :param encoder: The :class:`~dcba.models.config_encoder.ConfigEncoder` to train.
    :param optimizer_config: AdamW hyperparameters dict, expected keys ``lr`` and
        ``weight_decay``.
    """

    def __init__(self, encoder: ConfigEncoder, optimizer_config: dict) -> None:
        """Initialise the wrapper with the encoder and optimiser settings."""
        super().__init__()
        self._encoder = encoder
        self._optimizer_config = optimizer_config

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run the encoder forward pass.

        :param x: Input config tensor of shape ``(batch, input_dim)``.

        :returns: Tuple ``(h_q, x_hat)``.
        """
        return self._encoder(x)

    def _step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        x, target = batch
        _, x_hat = self._encoder(x)
        loss = F.mse_loss(x_hat, target)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        return self._step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Compute and log validation loss."""
        self._step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Compute and log test loss."""
        self._step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build and return an AdamW optimiser."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self._optimizer_config.get("lr", 1e-3),
            weight_decay=self._optimizer_config.get("weight_decay", 1e-5),
        )
