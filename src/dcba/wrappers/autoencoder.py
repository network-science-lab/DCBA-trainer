"""DCBAAutoencoderWrapper -- single-encoder training regime (autoencoder or regression)."""

from typing import cast
from unittest.mock import MagicMock

import torch.nn as nn
import wandb
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from dcba.dataset import ABCDConfigScaler
from dcba.dataset.transforms import ABCD_CONFIG_KEYS
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
    :param kl_loss: Optional KL divergence loss; only meaningful when ``encoder`` is a
        :class:`~dcba.models.config_vae.ConfigVAE`.
    :param beta_kl: Target weight for the KL term at steady state.
    :param kl_warmup_epochs: Epochs over which ``beta_kl`` is linearly annealed from 0 to its
        target value.  Set to 0 to use a constant ``beta_kl`` from step 0.
    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer_config: dict,
        scaler: ABCDConfigScaler | None = None,
        loss_fn: nn.Module | None = None,
        kl_loss: nn.Module | None = None,
        beta_kl: float = 1.0,
        kl_warmup_epochs: int = 0,
    ) -> None:
        """Initialise the wrapper with the encoder, optimiser settings, and optional losses."""
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "loss_fn", "kl_loss"])
        self._encoder = encoder
        self._optimizer_config = optimizer_config
        self._scaler = scaler
        self._loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self._kl_loss = kl_loss
        self._beta_kl = beta_kl
        self._kl_warmup_epochs = kl_warmup_epochs
        self._test_rows: list[tuple[list[float], list[float]]] = []

    def forward(self, batch: DCBAHeteroData) -> ForwardOutput:
        """
        Run the encoder forward pass.

        :param batch: Batched heterogeneous graph data.

        :returns: :class:`~dcba.models.types.ForwardOutput` from the wrapped encoder.
        """
        return self._encoder(batch)

    def _step(self, batch: DCBAHeteroData, stage: str) -> tuple[Tensor]:
        config = self._unpack_batch(batch)
        out = self._encoder(batch)
        loss = self._loss_fn(out.reconstruction, config)
        if self._kl_loss is not None:
            loss = loss + self._kl_term(self._encoder, config, batch.batch_size, stage)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch.batch_size)
        return (loss,)

    def test_step(
        self,
        batch: DCBAHeteroData,
        batch_idx: int,
    ) -> None:
        """Compute and log test loss; accumulate per-sample reconstruction rows."""
        config = self._unpack_batch(batch)
        out = self._encoder(batch)
        loss = self._loss_fn(out.reconstruction, config)
        self.log("test_loss", loss, prog_bar=True, batch_size=batch.batch_size)

        config_cpu = config.detach().cpu()
        recon_cpu = out.reconstruction.detach().cpu()

        if self._scaler is not None:
            config_cpu = self._scaler.inverse_transform(config_cpu)
            recon_cpu = self._scaler.inverse_transform(recon_cpu)

        for orig, recon in zip(config_cpu.tolist(), recon_cpu.tolist(), strict=True):
            self._test_rows.append((cast(list[float], orig), cast(list[float], recon)))

    def on_test_epoch_end(self) -> None:
        """Log original and reconstructed configs as a single wandb Table.

        Each sample occupies two consecutive rows distinguished by a ``sample`` label
        of the form ``{i}-o`` (original) and ``{i}-r`` (reconstructed), so wandb's
        table controls can filter and compare them side by side.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        suffix = "" if self._scaler is not None else "_norm"
        columns = ["sample"] + [f"{k}{suffix}" for k in ABCD_CONFIG_KEYS]
        rows = []
        for i, (orig, recon) in enumerate(self._test_rows):
            rows.append([f"{i}-o"] + orig)
            rows.append([f"{i}-r"] + recon)
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/reconstructions": table})
