"""Generic dataset for DCBA configuration records."""

from pathlib import Path

from dcba_data_set.graph_io import load_report
from dcba_data_set.graph_io.data_models import ConfigRecord
from torch import Tensor
from torch.utils.data import Dataset

from dcba.dataset.transforms import ABCDConfigScaler, ABCDConfigToTensor


class ConfigDataset(Dataset):
    """
    Dataset of configuration tensors for autoencoder training.

    Each item is a ``(tensor, tensor)`` pair where both elements are identical —
    the input and reconstruction target for the autoencoder.

    Config records are converted to tensors via
    :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.  An optional
    :class:`~dcba.dataset.transforms.ABCDConfigScaler` is applied afterwards to normalise each
    feature to ``[0, 1]``.

    :param configs: Mapping of ``instance_id`` to :class:`ConfigRecord`, as returned by
        :func:`~dcba_data_set.graph_io.load_report`.
    :param scaler: Optional scaler applied after the tensor transform to normalise features.
        When ``None``, tensors are returned in their raw (unscaled) form.
    """

    def __init__(
        self,
        configs: dict[str, ConfigRecord],
        scaler: ABCDConfigScaler | None = None,
    ) -> None:
        """Initialise the dataset, eagerly converting records to tensors and applying the scaler."""
        super().__init__()
        to_tensor = ABCDConfigToTensor()
        self._tensors: list[Tensor] = [
            scaler(to_tensor(r)) if scaler is not None else to_tensor(r) for r in configs.values()
        ]

    @classmethod
    def from_report(
        cls,
        report_path: Path,
        scaler: ABCDConfigScaler | None = None,
    ) -> "ConfigDataset":
        """
        Build a dataset by loading a report.json manifest.

        :param report_path: Path to the ``report.json`` produced by
            :class:`~dcba_data_set.ds_generator.DatasetGenerator`.
        :param scaler: Optional scaler applied after the tensor transform.

        :returns: A :class:`ConfigDataset` constructed from all configs in the report.
        """
        configs, _ = load_report(report_path)
        return cls(configs, scaler=scaler)

    def __len__(self) -> int:
        """Return the number of config instances in the dataset."""
        return len(self._tensors)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Return the config tensor at the given index as an (input, target) pair.

        :param idx: Index of the config instance.

        :returns: A tuple ``(input, target)`` of independent float tensors of shape ``(9,)``, both
            containing the same values.
        """
        t = self._tensors[idx]
        return t, t.clone()
