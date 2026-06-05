"""Base LightningModule for DCBA training regimes."""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor

from dcba.dataset import ABCDConfigScaler


class KLRegularisedMixin:
    """
    Mixin that adds beta-annealed KL regularisation for ConfigVAE-based config encoders.

    Designed for :class:`~dcba.models.config_vae.ConfigVAE`-based config encoders.

    Concrete subclasses must declare and initialise ``_kl_loss``, ``_beta_kl``,
    ``_kl_warmup_epochs``, and ``_kl_warmup_steps`` in their own ``__init__``.
    They also inherit ``self.trainer``, ``self.global_step``, and ``self.log``
    from :class:`~lightning.pytorch.LightningModule` via the main base class.
    """

    _kl_loss: nn.Module | None
    _beta_kl: float
    _kl_warmup_epochs: int
    _kl_warmup_steps: int

    def on_train_start(self) -> None:
        """Convert KL warmup epochs to steps using the actual number of batches per epoch."""
        if self._kl_loss is not None and self._kl_warmup_epochs > 0:
            self._kl_warmup_steps = self._kl_warmup_epochs * self.trainer.num_training_batches

    def _kl_term(self, mu: Tensor, log_sigma: Tensor, batch_size: int, stage: str) -> Tensor:
        """
        Compute the beta-annealed KL term and log raw KL and effective beta.

        :param mu: Posterior mean of shape ``(batch, embedding_dim)``, from
            :meth:`~dcba.models.config_vae.ConfigVAE.encode_distribution`.
        :param log_sigma: Posterior log standard deviation, same shape as ``mu``.
        :param batch_size: Batch size for Lightning logging.
        :param stage: One of ``"train"``, ``"val"``, ``"test"``.

        :returns: Scalar ``effective_beta * KL``, ready to add to the total loss.
        """
        l_kl = self._kl_loss(mu, log_sigma)  # type: ignore[misc]
        # Linear warmup: beta rises from 0 to beta_kl over kl_warmup_steps training steps so that
        # SupCon / reconstruction can establish a signal before the prior constraint kicks in.
        effective_beta = (
            self._beta_kl * min(1.0, self.global_step / self._kl_warmup_steps)
            if self._kl_warmup_steps > 0
            else self._beta_kl
        )
        self.log(f"{stage}_loss-kl", l_kl, batch_size=batch_size)
        self.log(f"{stage}_beta-kl", effective_beta, batch_size=batch_size)
        return effective_beta * l_kl


class DCBABaseWrapper(pl.LightningModule):
    """
    Abstract base for DCBA LightningModule wrappers.

    Provides shared boilerplate: batch unpacking, train/val step dispatch,
    scaler management, test-row reset, and AdamW optimiser construction.
    Subclasses must implement :meth:`_step`.
    """

    def __init__(self) -> None:
        """Initialise shared instance attributes to their defaults."""
        super().__init__()
        self._optimizer_config: dict = {}
        self._scaler: ABCDConfigScaler | None = None
        self._test_rows: list = []

    def _unpack_batch(self, batch: DCBAHeteroData) -> Tensor:
        """Reshape and return the config tensor; used as encoder input and regression target."""
        return batch.config.reshape(batch.batch_size, -1)

    def _step(self, batch: DCBAHeteroData, stage: str) -> tuple[Tensor, ...]:
        raise NotImplementedError

    def training_step(self, batch: DCBAHeteroData, batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        return self._step(batch, "train")[0]

    def validation_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Compute and log validation loss."""
        self._step(batch, "val")

    def on_test_epoch_start(self) -> None:
        """Reset the per-sample accumulator."""
        self._test_rows = []

    def set_scaler(self, scaler: ABCDConfigScaler | None) -> None:
        """Set the scaler used to inverse-transform tensors before test logging."""
        self._scaler = scaler

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Build and return an AdamW optimiser."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=self._optimizer_config.get("lr", 1e-3),
            weight_decay=self._optimizer_config.get("weight_decay", 1e-5),
        )
