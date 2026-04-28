"""DCBASupConWrapper -- joint graph + config encoder training with SupCon + regression loss."""

from typing import cast
from unittest.mock import MagicMock

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from dcba.dataset import ABCDConfigScaler
from dcba.dataset.transforms import ABCD_CONFIG_KEYS
from dcba.training.loss import MultiPositiveSupConLoss


class DCBASupConWrapper(pl.LightningModule):
    """
    LightningModule for joint training of graph and config encoders.

    Optimises ``L = L_reg + lambda * L_SupCon`` where:

    - ``L_reg`` -- MSE between the graph encoder's predicted ``theta_hat`` and ground-truth
      ``theta``.
    - ``L_SupCon`` -- :class:`~dcba.training.loss.MultiPositiveSupConLoss` with ``h_G`` as anchors
      and ``h_theta`` providing cross-modal positives and negatives.

    Both encoders are trained jointly.  The config encoder's reconstruction output is not used.
    Logs ``{stage}_loss``, ``{stage}_l_reg``, and ``{stage}_l_supcon`` at every step.

    :param graph_encoder: GNN that produces ``(h_G, theta_hat)`` -- in practice
        :class:`~dcba.models.ff_graph_config_predictor.FeedforwardGraphConfigPredictor`.
    :param config_encoder: MLP autoencoder that produces ``(h_theta, theta_hat)`` -- in practice
        :class:`~dcba.models.config_autoencoder.ConfigAutoEncoder`.
    :param optimizer_config: AdamW hyperparameters dict, expected keys ``lr`` and ``weight_decay``.
    :param lambda_supcon: Weight ``lambda`` applied to the contrastive loss term.
    :param supcon_loss: Pre-built :class:`~dcba.training.loss.MultiPositiveSupConLoss` instance.
    :param scaler: Optional scaler for inverse-transforming tensors before test logging.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        config_encoder: nn.Module,
        optimizer_config: dict,
        supcon_loss: MultiPositiveSupConLoss,
        lambda_supcon: float = 1.0,
        scaler: ABCDConfigScaler | None = None,
    ) -> None:
        """Initialise with both encoders, optimiser settings, and loss hyperparameters."""
        super().__init__()
        self.save_hyperparameters(ignore=["graph_encoder", "config_encoder", "supcon_loss"])
        self._graph_encoder = graph_encoder
        self._config_encoder = config_encoder
        self._optimizer_config = optimizer_config
        self._lambda_supcon = lambda_supcon
        self._supcon_loss = supcon_loss
        self._scaler = scaler
        self._test_rows: list[tuple[list[float], list[float]]] = []

    def _unpack_batch(self, batch: DCBAHeteroData) -> tuple[Tensor, DCBAHeteroData, Tensor]:
        config = batch.config.reshape(batch.batch_size, -1)
        target = batch.y.reshape(batch.batch_size, -1)
        return config, batch, target

    @staticmethod
    def _instance_labels(instance_ids: list[str], device: torch.device) -> Tensor:
        """
        Convert a list of ``instance_id`` strings into a compact integer label tensor.

        :param instance_ids: Per-sample instance identifiers as produced by PyG collation.
        :param device: Target device for the output tensor.

        :returns: ``(B,)`` int64 tensor of contiguous group indices.
        """
        uid_to_idx = {uid: idx for idx, uid in enumerate(sorted(set(instance_ids)))}
        return torch.tensor(
            [uid_to_idx[uid] for uid in instance_ids], dtype=torch.long, device=device
        )

    def _step(self, batch: DCBAHeteroData, stage: str) -> Tensor:
        config, graph, target = self._unpack_batch(batch)

        h_g, theta_hat = self._graph_encoder((config, graph))
        h_theta, _ = self._config_encoder((config, graph))

        labels = self._instance_labels(batch.instance_id, device=h_g.device)

        l_reg = F.mse_loss(theta_hat, target)
        l_supcon = self._supcon_loss(h_g, h_theta, labels, config)
        loss = l_reg + self._lambda_supcon * l_supcon

        bs = graph.batch_size
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=bs)
        self.log(f"{stage}_l_reg", l_reg, batch_size=bs)
        self.log(f"{stage}_l_supcon", l_supcon, batch_size=bs)
        return loss

    def training_step(self, batch: DCBAHeteroData, batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        return self._step(batch, "train")

    def validation_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Compute and log validation loss."""
        self._step(batch, "val")

    def on_test_epoch_start(self) -> None:
        """Reset the per-sample prediction accumulator."""
        self._test_rows = []

    def test_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Compute and log test loss; accumulate graph-encoder predictions for logging."""
        self._step(batch, "test")

        config, graph, _ = self._unpack_batch(batch)
        with torch.no_grad():
            _, theta_hat = self._graph_encoder((config, graph))

        x_cpu = config.detach().cpu()
        x_hat_cpu = theta_hat.detach().cpu()

        if self._scaler is not None:
            x_cpu = self._scaler.inverse_transform(x_cpu)
            x_hat_cpu = self._scaler.inverse_transform(x_hat_cpu)

        for orig, pred in zip(x_cpu.tolist(), x_hat_cpu.tolist(), strict=True):
            self._test_rows.append((cast(list[float], orig), cast(list[float], pred)))

    def on_test_epoch_end(self) -> None:
        """Log original configs and graph-encoder predictions as a wandb Table.

        Rows labelled ``{i}-o`` (original) and ``{i}-p`` (predicted) for side-by-side comparison.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        suffix = "" if self._scaler is not None else "_norm"
        columns = ["sample"] + [f"{k}{suffix}" for k in ABCD_CONFIG_KEYS]
        rows = []
        for i, (orig, pred) in enumerate(self._test_rows):
            rows.append([f"{i}-o"] + orig)
            rows.append([f"{i}-p"] + pred)
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/predictions": table})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build and return an AdamW optimiser over both encoders."""
        return torch.optim.AdamW(
            list(self._graph_encoder.parameters()) + list(self._config_encoder.parameters()),
            lr=self._optimizer_config.get("lr", 1e-3),
            weight_decay=self._optimizer_config.get("weight_decay", 1e-5),
        )
