"""PyTorch dataset for DCBA config autoencoder training."""

from pathlib import Path

import torch
from dcba_data_set.graph_io import load_report
from dcba_data_set.graph_io.data_models import ConfigRecord
from torch import Tensor
from torch.utils.data import Dataset

#: Ordered list of numerical ABCD config keys used as model input features.
_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]


def _config_to_tensor(record: ConfigRecord) -> Tensor:
    """
    Extract the 9 numerical ABCD config fields from a ConfigRecord as a float tensor.

    :param record: A loaded ABCD config record.

    :returns: A 1-D float tensor of shape ``(9,)`` with values in the order defined by
        ``_CONFIG_KEYS``.
    """
    values = [float(record.data[key]) for key in _CONFIG_KEYS]
    return torch.tensor(values, dtype=torch.float32)


class DCBAConfigDataset(Dataset):
    """
    Dataset of ABCD configuration tensors for autoencoder training.

    Each item is a ``(tensor, tensor)`` pair where both elements are identical —
    the input and reconstruction target for the autoencoder.

    Instances are identified by ``instance_id``; all replicas of an instance appear in
    the same split to avoid data leakage between training and validation/test sets.

    :param configs: Mapping of ``instance_id`` to :class:`ConfigRecord`, as returned by
        :func:`~dcba_data_set.graph_io.load_report`.
    """

    def __init__(self, configs: dict[str, ConfigRecord]) -> None:
        """Initialise the dataset from a mapping of instance_id to ConfigRecord."""
        super().__init__()
        self._tensors: list[Tensor] = [_config_to_tensor(r) for r in configs.values()]

    @classmethod
    def from_report(cls, report_path: Path) -> "DCBAConfigDataset":
        """
        Build a dataset by loading a report.json manifest.

        :param report_path: Path to the ``report.json`` produced by
            :class:`~dcba_data_set.ds_generator.DatasetGenerator`.

        :returns: A :class:`DCBAConfigDataset` constructed from all configs in the report.
        """
        configs, _ = load_report(report_path)
        return cls(configs)

    def __len__(self) -> int:
        """Return the number of config instances in the dataset."""
        return len(self._tensors)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Return the config tensor at the given index as an (input, target) pair.

        :param idx: Index of the config instance.

        :returns: A tuple ``(tensor, tensor)`` where both elements are the same float
            tensor of shape ``(9,)``.
        """
        t = self._tensors[idx]
        return t, t
