"""Generic dataset for DCBA configuration records."""

from pathlib import Path

from dcba_data_set.graph_io import load_report
from dcba_data_set.graph_io.data_models import ConfigRecord
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.transforms import BaseTransform

from dcba.dataset.transforms import ABCDConfigToTensor


class ConfigDataset(Dataset):
    """
    Dataset of configuration tensors for autoencoder training.

    Each item is a ``(tensor, tensor)`` pair where both elements are identical —
    the input and reconstruction target for the autoencoder.

    Instances are identified by ``instance_id``; all replicas of an instance appear in
    the same split to avoid data leakage between training and validation/test sets.

    The transform determines which config keys are extracted and how they are encoded,
    making this class reusable for both ABCD and mABCD configs.

    :param configs: Mapping of ``instance_id`` to :class:`ConfigRecord`, as returned by
        :func:`~dcba_data_set.graph_io.load_report`.
    :param transform: Transform applied to each :class:`ConfigRecord` to produce a tensor.
        Defaults to :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.
    """

    def __init__(
        self,
        configs: dict[str, ConfigRecord],
        transform: BaseTransform | None = None,
    ) -> None:
        """Initialise the dataset from a mapping of instance_id to ConfigRecord."""
        super().__init__()
        if transform is None:
            transform = ABCDConfigToTensor()
        self._tensors: list[Tensor] = [transform(r) for r in configs.values()]

    @classmethod
    def from_report(
        cls,
        report_path: Path,
        transform: BaseTransform | None = None,
    ) -> "ConfigDataset":
        """
        Build a dataset by loading a report.json manifest.

        :param report_path: Path to the ``report.json`` produced by
            :class:`~dcba_data_set.ds_generator.DatasetGenerator`.
        :param transform: Transform applied to each config record. Defaults to
            :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.

        :returns: A :class:`ConfigDataset` constructed from all configs in the report.
        """
        configs, _ = load_report(report_path)
        return cls(configs, transform=transform)

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
