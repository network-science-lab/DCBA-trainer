"""PyTorch Lightning DataModule for DCBA config autoencoder training."""

from pathlib import Path

import lightning.pytorch as pl
from dcba_data_set.graph_io import load_report
from torch.utils.data import DataLoader

from dcba.dataset import ConfigDataset


class DCBADataModule(pl.LightningDataModule):
    """
    LightningDataModule that loads an ABCD report and splits it into train/val/test sets.

    Splitting is performed on ``instance_id`` keys to prevent data leakage between splits.

    :param report_path: Path to the ``report.json`` manifest.
    :param val_ratio: Fraction of instances to use for validation.
    :param test_ratio: Fraction of instances to use for testing.
    :param batch_size: Number of samples per dataloader batch.
    :param num_workers: Number of worker processes for the dataloaders.
    """

    def __init__(
        self,
        report_path: Path,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        """Initialise the data module with dataset path and split/loader parameters."""
        super().__init__()
        self._report_path = Path(report_path)
        self._val_ratio = val_ratio
        self._test_ratio = test_ratio
        self._batch_size = batch_size
        self._num_workers = num_workers

        self._train_dataset: ConfigDataset | None = None
        self._val_dataset: ConfigDataset | None = None
        self._test_dataset: ConfigDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Load the report and build train/val/test splits.

        :param stage: Lightning stage identifier (``"fit"``, ``"test"``, etc.).
            Unused here — all splits are always prepared.
        """
        configs, _ = load_report(self._report_path)
        instance_ids = list(configs.keys())
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

        self._train_dataset = ConfigDataset({k: configs[k] for k in train_ids})
        self._val_dataset = ConfigDataset({k: configs[k] for k in val_ids})
        self._test_dataset = ConfigDataset({k: configs[k] for k in test_ids})

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
