"""PyTorch Lightning DataModule for DCBA config autoencoder training."""

import random
from pathlib import Path

import lightning.pytorch as pl
from dcba_data_set.graph_io import load_dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform

from dcba.dataset import ABCDConfigScaler, ABCDDataset


class ABCDDataModule(pl.LightningDataModule):
    """
    LightningDataModule that loads an ABCD report and splits it into train/val/test sets.

    Splitting is performed on ``instance_id`` keys to prevent data leakage between splits.
    The ``scaler`` and ``transform`` are injected by the caller (typically the training entry
    point) and stored as public attributes so wrappers can retrieve them at inference time.

    :param dataset_root: Root directory of the dataset (flat or chunked layout).
    :param val_ratio: Fraction of instances to use for validation.
    :param test_ratio: Fraction of instances to use for testing.
    :param batch_size: Number of samples per dataloader batch.
    :param num_workers: Number of worker processes for the dataloaders.
    :param single_replica_per_instance: When ``True``, only the first replica per instance is
        used.  When ``False``, all replicas are included.
    :param seed: RNG seed used to shuffle instances before splitting.  Fixes the train/val/test
        assignment across runs.
    :param scaler: Optional scaler applied after the transform to normalise config features.
        When ``None``, configs are kept in raw form.
    :param transform: Transform applied to each config record.  Defaults to
        :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.
    """

    def __init__(
        self,
        dataset_root: Path,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        single_replica_per_instance: bool = True,
        seed: int = 42,
        scaler: ABCDConfigScaler | None = None,
        transform: BaseTransform | None = None,
    ) -> None:
        """Initialise the data module with dataset path and split/loader parameters."""
        super().__init__()
        self._dataset_root = Path(dataset_root)
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._single_replica_per_instance = single_replica_per_instance
        self._seed = seed

        self.scaler: ABCDConfigScaler | None = scaler
        self.transform: BaseTransform | None = transform
        self._train_dataset: ABCDDataset | None = None
        self._val_dataset: ABCDDataset | None = None
        self._test_dataset: ABCDDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Load the report and build train/val/test splits.

        :param stage: Lightning stage identifier (``"fit"``, ``"test"``, etc.).
            Unused here — all splits are always prepared.
        """
        records = load_dataset(self._dataset_root)
        random.Random(self._seed).shuffle(records)
        instance_ids = [r.instance_id for r in records]
        n = len(instance_ids)

        n_test = max(1, int(n * self._test_ratio))
        n_val = max(1, int(n * self._val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            raise ValueError(
                f"Not enough instances ({n}) for val_ratio={self._val_ratio} and "
                f"test_ratio={self._test_ratio}."
            )

        train_ids = set(instance_ids[:n_train])
        val_ids = set(instance_ids[n_train : n_train + n_val])
        test_ids = set(instance_ids[n_train + n_val :])

        self._train_dataset = ABCDDataset(
            records=[r for r in records if r.instance_id in train_ids],
            scaler=self.scaler,
            transform=self.transform,
            single_replica_per_instance=self._single_replica_per_instance,
        )
        self._val_dataset = ABCDDataset(
            records=[r for r in records if r.instance_id in val_ids],
            scaler=self.scaler,
            transform=self.transform,
            single_replica_per_instance=self._single_replica_per_instance,
        )
        self._test_dataset = ABCDDataset(
            records=[r for r in records if r.instance_id in test_ids],
            scaler=self.scaler,
            transform=self.transform,
            single_replica_per_instance=self._single_replica_per_instance,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
        )
