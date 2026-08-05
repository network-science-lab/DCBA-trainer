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

from dcba.dataset import ABCDBaseConfigScaler
from dcba.dataset.scalers import ABCD_CONFIG_KEYS
from dcba.eval.predictions_table import build_predictions_table
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
    Logs ``{stage}_loss``, ``{stage}_loss-reg-t``, ``{stage}_loss-reg-g``, and
    ``{stage}_loss-contr`` at every step.

    :param graph_encoder: GNN that produces ``(z_g, theta_hat)`` -- in practice
        :class:`~dcba.models.gin_encoder.GINEncoder`.
    :param config_encoder: MLP encoder that produces ``(z_theta, theta_hat)`` -- in practice
        :class:`~dcba.models.config_ae.ConfigAutoEncoder`.
    :param optimizer_config: AdamW hyperparameters dict, expected keys ``lr`` and ``weight_decay``.
    :param reg_loss: Loss module applied to ``(theta_hat, theta)`` for the regression term.
    :param supcon_loss: Pre-built :class:`~dcba.training.loss.MultiPositiveSupConLoss` instance.
    :param lambda_supcon: Weight ``lambda`` applied to the contrastive loss term.
    :param scaler: Optional scaler for inverse-transforming tensors before test logging.
    :param aux_weight: Weight applied to ``graph_encoder.aux_loss`` when present (e.g.
        :class:`~dcba.models.gps_encoder.GPSEncoder`'s cluster-assignment entropy loss, only set
        when that encoder's ``num_clusters > 0``). Ignored when the graph encoder does not expose
        an ``aux_loss`` attribute or leaves it ``None``.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        config_encoder: nn.Module,
        optimizer_config: dict,
        reg_loss: nn.Module,
        supcon_loss: nn.Module,
        lambda_supcon: float = 1.0,
        scaler: ABCDBaseConfigScaler | None = None,
        aux_weight: float = 0.1,
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
        self._aux_weight = aux_weight
        self._scaler = scaler
        self._test_rows: list[tuple[str, int, list[float], list[float], list[float]]] = []

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

    @torch.no_grad()
    def encode(self, batch: DCBAHeteroData) -> tuple[Tensor, Tensor]:
        """
        Compute unit-norm ``(z_g, z_theta)`` embeddings for a batch, without loss or logging.

        For analysis/inference use (e.g. embedding-stability scripts) where only the encoder
        outputs are needed.

        :param batch: Batched heterogeneous graph data with ``.config`` attached.

        :returns: ``(z_g, z_theta)``, each ``(batch, embedding_dim)`` and unit-norm.
        """
        config = self._unpack_batch(batch)
        z_g = F.normalize(self._graph_encoder.encode(batch), dim=-1)
        z_theta = F.normalize(self._config_encoder.encode(config), dim=-1)
        return z_g, z_theta

    @torch.no_grad()
    def encode_and_reconstruct(self, batch: DCBAHeteroData) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute ``(theta, theta_hat_regr, theta_hat_cross)`` for a batch, without loss or logging.

        Same computation as :meth:`_step`'s regression path, minus the loss terms and ``self.log``
        calls -- for offline analysis scripts (e.g. ``scripts/compute_test_predictions.py``) that
        call this outside a wired :class:`~lightning.pytorch.Trainer`, where ``self.log`` would
        raise.

        :param batch: Batched heterogeneous graph data with ``.config`` attached.

        :returns: ``theta`` (ground truth), ``theta_hat_regr`` (config-encoder self-reconstruction),
            and ``theta_hat_cross`` (graph-encoder -> theta-decoder cross-modal prediction).
        """
        config = self._unpack_batch(batch)
        z_g, z_theta = self.encode(batch)
        theta_hat_zt = self._config_encoder.decode(z_theta)
        theta_hat_zg = self._config_encoder.decode(z_g)
        return config, theta_hat_zt, theta_hat_zg

    def _step(self, batch: DCBAHeteroData, stage: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        config = self._unpack_batch(batch)
        batch_size = cast(int, batch.batch_size)

        z_g = F.normalize(self._graph_encoder.encode(batch), dim=-1)
        z_theta = F.normalize(self._config_encoder.encode(config), dim=-1)

        theta_hat_zt = self._config_encoder.decode(z_theta)
        theta_hat_zg = self._config_encoder.decode(z_g)

        labels = self._instance_labels(cast(list[str], batch.instance_id), device=z_g.device)

        l_reg_zt = self._reg_loss(theta_hat_zt, config)
        l_reg_zg = self._reg_loss(theta_hat_zg, config)
        l_supcon = self._supcon_loss(z_g, z_theta, labels, config)
        loss = l_reg_zg + l_reg_zt + self._lambda_supcon * l_supcon

        aux_loss = getattr(self._graph_encoder, "aux_loss", None)
        if aux_loss is not None:
            loss = loss + self._aux_weight * aux_loss
            self.log(f"{stage}_loss-cluster-entropy", aux_loss, batch_size=batch_size)
            # Logged separately from aux_loss because usage_entropy_weight=0 drops it from the
            # loss (see GPSEncoder.usage_entropy for what exp(value) means).
            usage_entropy = getattr(self._graph_encoder, "usage_entropy", None)
            if usage_entropy is not None:
                self.log(
                    f"{stage}_loss-cluster-usage-entropy", usage_entropy, batch_size=batch_size
                )
            # Diagnostics rather than loss components (see GPSEncoderShape). The cut is the
            # fraction of edge volume kept inside a slot, and `1 - cut` tracks xi. Modularity says
            # whether the learned slots are communities at all: ground-truth ABCD partitions score
            # ~0.41, so a value near 0 means the size and density channels describe nothing.
            for name, attribute in (("cut", "normalised_cut"), ("modularity", "modularity")):
                value = getattr(self._graph_encoder, attribute, None)
                if value is not None:
                    self.log(f"{stage}_loss-cluster-{name}", value, batch_size=batch_size)

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_loss-reg-t", l_reg_zt, batch_size=batch_size)
        self.log(f"{stage}_loss-reg-g", l_reg_zg, batch_size=batch_size)
        self.log(f"{stage}_loss-contr", l_supcon, batch_size=batch_size)
        return loss, config, theta_hat_zt, theta_hat_zg

    def test_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Compute and log test loss; accumulate per-sample rows for the prediction table."""
        loss, config, theta_hat_zt, theta_hat_zg = self._step(batch, "test")

        config_cpu = config.detach().cpu()
        recon_cpu = theta_hat_zt.detach().cpu()
        cross_cpu = theta_hat_zg.detach().cpu()
        replicas = batch.replica.detach().cpu()

        if self._scaler is not None:
            config_cpu = self._scaler.inverse_transform(config_cpu)
            recon_cpu = self._scaler.inverse_transform(recon_cpu)
            cross_cpu = self._scaler.inverse_transform(cross_cpu)

        for instance_id, replica, orig, recon, cross in zip(
            cast(list[str], batch.instance_id),
            replicas.tolist(),
            config_cpu.tolist(),
            recon_cpu.tolist(),
            cross_cpu.tolist(),
            strict=True,
        ):
            self._test_rows.append(
                (
                    instance_id,
                    replica,
                    cast(list[float], orig),
                    cast(list[float], recon),
                    cast(list[float], cross),
                )
            )

    def on_test_epoch_end(self) -> None:
        """
        Log a wandb Table with original θ, config reconstruction, and cross-modal predictions.

        Row layout built by :func:`~dcba.eval.predictions_table.build_predictions_table`, shared
        with ``scripts/compute_test_predictions.py``'s local (non-wandb) equivalent.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        columns, rows = build_predictions_table(
            self._test_rows, ABCD_CONFIG_KEYS, has_scaler=self._scaler is not None
        )
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/predictions": table})
