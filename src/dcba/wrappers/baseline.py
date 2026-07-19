"""DCBABaselineWrapper -- heuristic ABCD baseline, evaluated on the test split only."""

from typing import cast
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
import wandb
from dcba_data_set.baseline import BaselineConfig
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from lightning.pytorch.loggers import WandbLogger

from dcba.dataset.transforms import ABCD_CONFIG_KEYS
from dcba.wrappers.base import DCBABaseWrapper


class DCBABaselineWrapper(DCBABaseWrapper):
    """
    LightningModule wrapping the heuristic :class:`~dcba_data_set.baseline.BaselineConfig`.

    Recovers ABCD generator parameters directly from graph structure -- there are no learned
    parameters, so there is nothing to train. ``training_step``/``validation_step`` are no-ops
    and ``configure_optimizers`` returns ``None`` (the officially supported "fit without an
    optimizer" mode -- see :meth:`~lightning.pytorch.core.LightningModule.configure_optimizers`).
    Callers must still pair this wrapper with ``training.max_epochs: 0`` and
    ``training.num_sanity_val_steps: 0`` in config so ``Trainer.fit`` never touches a batch;
    only ``Trainer.test`` performs real work.

    Requires ``data.batch_size: 1``: :class:`~dcba_data_set.baseline.BaselineConfig` computes
    per-graph structural statistics (communities, degrees) and does not support a batch of
    several disjoint graphs collated into one.

    :param detect_communities: If True, communities are detected via the Leiden algorithm
        instead of using ground-truth labels stored on the graph.
    :param leiden_resolution: Resolution parameter for Leiden; only used when
        ``detect_communities`` is True.
    :param leiden_seed: Random seed for Leiden; only used when ``detect_communities`` is True.
    :param d_max_iter: Maximum degree-sampling iterations, passed through to the resulting
        ``ABCDConfig``.
    :param c_max_iter: Maximum community-sampling iterations, passed through to the resulting
        ``ABCDConfig``.
    :param natural_cutoff: If True, estimate d_max and c_max using the natural cutoff formula
        instead of the raw sample maximum, which underestimates the true upper bound of a
        truncated power-law sample.
    """

    def __init__(
        self,
        detect_communities: bool = False,
        leiden_resolution: float = 1.0,
        leiden_seed: int | None = None,
        d_max_iter: int = 1000,
        c_max_iter: int = 1000,
        natural_cutoff: bool = False,
    ) -> None:
        """Initialise the wrapper with the BaselineConfig hyperparameters."""
        super().__init__()
        self.save_hyperparameters()
        self._detect_communities = detect_communities
        self._leiden_resolution = leiden_resolution
        self._leiden_seed = leiden_seed
        self._d_max_iter = d_max_iter
        self._c_max_iter = c_max_iter
        self._natural_cutoff = natural_cutoff
        self._test_rows: list[tuple[str, int, list[float], list[float]]] = []

    def training_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """No-op: the baseline has no learnable parameters."""
        return None

    def validation_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """No-op: the baseline has no learnable parameters."""
        return None

    def configure_optimizers(self) -> None:
        """Return None so Trainer.fit runs without an optimiser."""
        return None

    def test_step(self, batch: DCBAHeteroData, batch_idx: int) -> None:
        """Extract the heuristic config from the graph and accumulate a prediction row."""
        batch_size = cast(int, batch.batch_size)
        if batch_size != 1:
            raise ValueError(
                f"DCBABaselineWrapper requires data.batch_size=1, got {batch_size}: "
                "BaselineConfig computes structural statistics per graph and cannot be batched."
            )

        predicted = BaselineConfig(
            batch,
            d_max_iter=self._d_max_iter,
            c_max_iter=self._c_max_iter,
            detect_communities=self._detect_communities,
            leiden_resolution=self._leiden_resolution,
            leiden_seed=self._leiden_seed,
            natural_cutoff=self._natural_cutoff,
        ).get_config()
        pred_values = [float(getattr(predicted, key)) for key in ABCD_CONFIG_KEYS]
        pred_tensor = torch.tensor(pred_values, dtype=torch.float32).unsqueeze(0)

        true_scaled = self._unpack_batch(batch).detach().cpu()
        if self._scaler is not None:
            true_raw = self._scaler.inverse_transform(true_scaled)
            pred_scaled = self._scaler.transform(pred_tensor)
        else:
            true_raw = true_scaled
            pred_scaled = pred_tensor

        loss = F.mse_loss(pred_scaled, true_scaled)
        self.log("test_loss", loss, batch_size=batch_size, prog_bar=True)

        instance_id = cast(list[str], batch.instance_id)[0]
        replica = int(batch.replica.detach().cpu().item())
        self._test_rows.append((instance_id, replica, true_raw[0].tolist(), pred_values))

    def on_test_epoch_end(self) -> None:
        """
        Log a wandb Table with the original theta and the baseline's predicted theta.

        Each sample occupies two consecutive rows: ``{i}-orig`` (ground-truth theta) and
        ``{i}-pred`` (BaselineConfig's heuristic prediction), matching the ``test/predictions``
        table layout used by the learned wrappers so results are directly comparable.
        """
        if not isinstance(self.logger, WandbLogger):
            return
        if isinstance(self.logger.experiment, MagicMock):
            return

        suffix = "" if self._scaler is not None else "_norm"
        columns = ["sample", "instance", "replica"] + [f"{k}{suffix}" for k in ABCD_CONFIG_KEYS]
        rows = []
        for i, (instance, replica, orig, pred) in enumerate(self._test_rows):
            rows.append([f"{i}-orig", instance, replica] + orig)
            rows.append([f"{i}-pred", instance, replica] + pred)
        table = wandb.Table(columns=columns, data=rows)
        self.logger.experiment.log({"test/predictions": table})
