"""DCBASupConWrapper -- joint graph + config encoder training with SupCon + regression loss."""

import hashlib
from typing import cast
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from dcba.dataset import ABCDConfigScaler
from dcba.dataset.transforms import ABCD_CONFIG_KEYS
from dcba.wrappers.base import DCBABaseWrapper


class DCBASupConWrapper(DCBABaseWrapper):
    """
    LightningModule for joint training of graph and config encoders.

    Optimises ``L = L_reg + lambda * L_SupCon`` where:

    - ``L_reg`` -- configurable regression loss (e.g. MSE or
      :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss`) applied to the config encoder's
      reconstructed ``theta_hat`` against ground-truth ``theta``.  Using the config encoder (a
      simple MLP) stabilises ``z_theta`` early in training, providing a stronger SupCon signal.
    - ``L_SupCon`` -- :class:`~dcba.training.loss.MultiPositiveSupConLoss` with ``z_g`` as anchors
      and ``z_theta`` providing cross-modal positives and negatives.

    Both encoders are trained jointly.
    Logs ``{stage}_loss``, ``{stage}_l_reg``, and ``{stage}_l_supcon`` at every step.

    :param graph_encoder: GNN that produces ``(z_g, theta_hat)`` -- in practice
        :class:`~dcba.models.gin_encoder.GINEncoder`.
    :param config_encoder: MLP autoencoder that produces ``(z_theta, theta_hat)`` -- in practice
        :class:`~dcba.models.config_autoencoder.ConfigAutoEncoder`.
    :param optimizer_config: AdamW hyperparameters dict, expected keys ``lr`` and ``weight_decay``.
    :param reg_loss: Loss module applied to ``(theta_hat, theta)`` for the regression term.
    :param supcon_loss: Pre-built :class:`~dcba.training.loss.MultiPositiveSupConLoss` instance.
    :param lambda_supcon: Weight ``lambda`` applied to the contrastive loss term.
    :param scaler: Optional scaler for inverse-transforming tensors before test logging.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        config_encoder: nn.Module,
        optimizer_config: dict,
        reg_loss: nn.Module,
        supcon_loss: nn.Module,
        lambda_supcon: float = 1.0,
        scaler: ABCDConfigScaler | None = None,
    ) -> None:
        """Initialise with both encoders, optimiser settings, and loss hyperparameters."""
        super().__init__()
        self.save_hyperparameters(
            ignore=["graph_encoder", "config_encoder", "reg_loss", "supcon_loss"]
        )
        self._graph_encoder = graph_encoder
        self._config_encoder = config_encoder
        self._optimizer_config = optimizer_config
        self._lambda_supcon = lambda_supcon
        self._reg_loss = reg_loss
        self._supcon_loss = supcon_loss
        self._scaler = scaler
        self._test_rows: list[tuple[list[float], list[float], list[float]]] = []

    @staticmethod
    def _instance_labels(instance_ids: list[str], device: torch.device) -> Tensor:
        """
        Convert a list of ``instance_id`` strings into globally consistent integer labels.

        Each string is hashed with SHA-256 so the same ``instance_id`` always maps to the
        same integer, regardless of what other samples are in the batch.  This is required
        for correctness when embeddings from different batches are concatenated (e.g. via a
        memory bank) -- per-batch compact re-indexing would make the same integer refer to
        different instances across steps, corrupting positive-pair detection.

        :param instance_ids: Per-sample instance identifiers as produced by PyG collation.
        :param device: Target device for the output tensor.

        :returns: ``(B,)`` int64 tensor of stable, globally unique group indices.
        """
        # SHA-256 handles any instance_id format; 15 hex chars = 60 bits, fits torch.long.
        return torch.tensor(
            [int(hashlib.sha256(uid.encode()).hexdigest()[:15], 16) for uid in instance_ids],
            dtype=torch.long,
            device=device,
        )

    def _step(self, batch: DCBAHeteroData, stage: str) -> Tensor:
        config = self._unpack_batch(batch)

        z_g = F.normalize(self._graph_encoder.encode(batch), dim=-1)
        z_theta = F.normalize(self._config_encoder.encode(config), dim=-1)
        theta_hat = self._config_encoder.decode(z_theta)
        theta_g_hat = self._config_encoder.decode(z_g)

        labels = self._instance_labels(cast(list[str], batch.instance_id), device=z_g.device)

        l_reg = self._reg_loss(theta_hat, config)
        l_reg_g = self._reg_loss(theta_g_hat, config)
        l_supcon = self._supcon_loss(z_g, z_theta, labels, config)
        loss = l_reg_g + l_reg + self._lambda_supcon * l_supcon

        batch_size = cast(int, batch.batch_size)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_l_reg", l_reg, batch_size=batch_size)
        self.log(f"{stage}_l_reg_g", l_reg_g, batch_size=batch_size)
        self.log(f"{stage}_l_supcon", l_supcon, batch_size=batch_size)
        return loss

    def test_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Compute and log test loss; accumulate per-sample rows for the prediction table."""
        self._step(batch, "test")

        config = self._unpack_batch(batch)
        z_g = F.normalize(self._graph_encoder.encode(batch), dim=-1)
        z_theta = F.normalize(self._config_encoder.encode(config), dim=-1)
        theta_hat_recon = self._config_encoder.decode(z_theta)
        theta_hat_cross = self._config_encoder.decode(z_g)

        config_cpu = config.detach().cpu()
        recon_cpu = theta_hat_recon.detach().cpu()
        cross_cpu = theta_hat_cross.detach().cpu()

        if self._scaler is not None:
            config_cpu = self._scaler.inverse_transform(config_cpu)
            recon_cpu = self._scaler.inverse_transform(recon_cpu)
            cross_cpu = self._scaler.inverse_transform(cross_cpu)

        for orig, recon, cross in zip(
            config_cpu.tolist(), recon_cpu.tolist(), cross_cpu.tolist(), strict=True
        ):
            self._test_rows.append(
                (cast(list[float], orig), cast(list[float], recon), cast(list[float], cross))
            )

    def on_test_epoch_end(self) -> None:
        """Log a wandb Table with original θ, config reconstruction, and cross-modal predictions.

        Each sample occupies three rows:

        - ``{i}-o``: original θ
        - ``{i}-r``: MLP reconstruction -- ``config_encoder.decode(config_encoder.encode(θ))``
        - ``{i}-c``: cross-modal prediction -- ``config_encoder.decode(graph_encoder.encode(G))``
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        suffix = "" if self._scaler is not None else "_norm"
        columns = ["sample"] + [f"{k}{suffix}" for k in ABCD_CONFIG_KEYS]
        rows = []
        for i, (orig, recon, cross) in enumerate(self._test_rows):
            rows.append([f"{i}-o"] + orig)
            rows.append([f"{i}-r"] + recon)
            rows.append([f"{i}-c"] + cross)
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/predictions": table})
