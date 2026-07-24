"""DCBAAutoencoderWrapper -- single-encoder training regime (autoencoder or regression)."""

from typing import cast
from unittest.mock import MagicMock

import torch.nn as nn
import wandb
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from dcba.dataset import ABCDConfigScaler
from dcba.dataset.scalers import ABCD_CONFIG_KEYS
from dcba.models.types import ForwardOutput
from dcba.wrappers.base import DCBABaseWrapper


class DCBAAutoencoderWrapper(DCBABaseWrapper):
    """
    LightningModule for Phase 1: trains the config autoencoder in isolation.

    Computes MSE reconstruction loss between the input config vector and the decoder
    output.  Logs ``{stage}_loss`` at every step.  During testing, accumulates
    per-sample reconstruction rows and logs them as a wandb Table.

    :param encoder: The :class: `nn.Module` to train. In practice it will be
        `~dcba.models.config_ae.ConfigAutoEncoder` or
        `~dcba.models.gin_encoder.GINEncoder`
    :param optimizer_config: AdamW hyperparameters dict, expected keys ``lr`` and
        ``weight_decay``.
    :param scaler: Optional scaler used to inverse-transform normalised tensors back
        to human-readable values before logging.  When ``None``, raw normalised values
        are logged with column names suffixed ``_norm``.
    :param loss_fn: Loss module used to compute the reconstruction error.
    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer_config: dict,
        scaler: ABCDConfigScaler | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        """Initialise the wrapper with the encoder, optimiser settings, and loss."""
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss_fn"])
        self._encoder = encoder
        self._optimizer_config = optimizer_config
        self._scaler = scaler
        self._loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self._test_rows: list[tuple[list[float], list[float]]] = []

    def forward(self, batch: DCBAHeteroData) -> ForwardOutput:
        """
        Run the encoder forward pass.

        :param batch: Batched heterogeneous graph data.

        :returns: :class:`~dcba.models.types.ForwardOutput` from the wrapped encoder.
        """
        return self._encoder(batch)

    def _step(self, batch: DCBAHeteroData, stage: str) -> tuple[Tensor, Tensor]:
        config = self._unpack_batch(batch)
        batch_size = cast(int, batch.batch_size)
        out = self._encoder(batch)
        loss = self._loss_fn(out.reconstruction, config)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        return loss, out.reconstruction

    def test_step(
        self,
        batch: DCBAHeteroData,
        batch_idx: int,
    ) -> None:
        """Compute and log test loss; accumulate per-sample reconstruction rows."""
        config = self._unpack_batch(batch)
        _, reconstruction = self._step(batch, "test")

        config_cpu = config.detach().cpu()
        recon_cpu = reconstruction.detach().cpu()

        if self._scaler is not None:
            config_cpu = self._scaler.inverse_transform(config_cpu)
            recon_cpu = self._scaler.inverse_transform(recon_cpu)

        for orig, recon in zip(config_cpu.tolist(), recon_cpu.tolist(), strict=True):
            self._test_rows.append((cast(list[float], orig), cast(list[float], recon)))

    def on_test_epoch_end(self) -> None:
        """Log original and reconstructed configs as a single wandb Table.

        Each sample occupies two consecutive rows: ``{i}-orig`` (original θ) and
        ``{i}-regr`` (reconstruction), so wandb's table controls can filter and
        compare them side by side.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        suffix = "" if self._scaler is not None else "_norm"
        columns = ["sample"] + [f"{k}{suffix}" for k in ABCD_CONFIG_KEYS]
        rows = []
        for i, (orig, recon) in enumerate(self._test_rows):
            rows.append([f"{i}-orig"] + orig)
            rows.append([f"{i}-regr"] + recon)
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/reconstructions": table})
