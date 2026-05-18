"""Base LightningModule for DCBA training regimes."""

import lightning.pytorch as pl
import torch
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor

from dcba.dataset import ABCDConfigScaler


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

    def _step(self, batch: DCBAHeteroData, stage: str) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: DCBAHeteroData, batch_idx: int) -> Tensor:
        """Compute and log training loss."""
        return self._step(batch, "train")

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
