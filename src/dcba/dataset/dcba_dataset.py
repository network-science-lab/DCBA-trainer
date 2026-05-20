"""Generic dataset for DCBA configuration records."""

import random
from pathlib import Path

from dcba_data_set.graph_io import load_dataset
from dcba_data_set.graph_io.data_models import (
    DCBAHeteroData,
    DCBAInstanceConfig,
    InstanceRecord,
    ReplicaRecord,
)
from torch.utils.data import Dataset
from torch_geometric.transforms import BaseTransform

from dcba.dataset.transforms import ABCDConfigScaler, ABCDConfigToTensor, CommunityToSize


class ABCDDataset(Dataset):
    """
    Dataset of configuration tensors & graphs for autoencoder training.

    Each item is a ``DCBAHeteroData`` containing the ABCD configuration record attached as
    ``.config`` and node features initialised to zeros.  Graphs are loaded lazily from disk on
    each ``__getitem__`` call.

    Config records are processed by ``transform`` (default:
    :class:`~dcba.dataset.transforms.ABCDConfigToTensor`).  An optional
    :class:`~dcba.dataset.transforms.ABCDConfigScaler` is applied afterwards when the transform
    produces a tensor, normalising each feature to ``[0, 1]``.

    :param records: List of :class:`~dcba_data_set.graph_io.data_models.InstanceRecord` objects,
        as returned by :func:`~dcba_data_set.graph_io.load_dataset`.
    :param scaler: Optional scaler applied after the transform.  When ``None``, config values are
        returned in their raw (unscaled) form.
    :param transform: Transform applied to each
        :class:`~dcba_data_set.graph_io.data_models.DCBAInstanceConfig` to produce the config
        representation stored in the dataset.  Defaults to
        :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.
    :param single_replica_per_instance: When ``True``, only the first replica of each instance
        is included.  When ``False``, all replicas are included.
    """

    def __init__(
        self,
        records: list[InstanceRecord],
        scaler: ABCDConfigScaler | None = None,
        transform: BaseTransform | None = None,
        single_replica_per_instance: bool = True,
    ) -> None:
        """Initialise the dataset, eagerly converting config records to tensors."""
        super().__init__()
        _transform = transform if transform is not None else ABCDConfigToTensor()
        self._items: list[tuple[ReplicaRecord, str, str, object]] = []

        for record in records:
            config = DCBAInstanceConfig.from_instance_record(record)
            c = scaler(_transform(config)) if scaler is not None else _transform(config)
            replica_list = record.replicas[:1] if single_replica_per_instance else record.replicas
            for replica in replica_list:
                self._items.append((replica, record.instance_id, record.net_type, c))

        random.shuffle(self._items)

    @classmethod
    def from_dataset(
        cls,
        dataset_root: Path,
        scaler: ABCDConfigScaler | None = None,
        transform: BaseTransform | None = None,
    ) -> "ABCDDataset":
        """
        Build a dataset from a dataset directory (flat or chunked layout).

        :param dataset_root: Root directory of the dataset, as produced by
            :class:`~dcba_data_set.ds_generator.DatasetGenerator`.
        :param scaler: Optional scaler applied after the transform.
        :param transform: Transform applied to each config record.  Defaults to
            :class:`~dcba.dataset.transforms.ABCDConfigToTensor`.

        :returns: A :class:`ABCDDataset` constructed from all records in the dataset.
        """
        return cls(records=load_dataset(dataset_root), scaler=scaler, transform=transform)

    def __len__(self) -> int:
        """Return the number of replica entries in the dataset."""
        return len(self._items)

    def __getitem__(self, idx: int) -> DCBAHeteroData:
        """
        Load and return the graph at the given index.

        :param idx: Index of the replica entry.

        :returns: A :class:`~dcba_data_set.graph_io.data_models.DCBAHeteroData` with the config
            representation attached as ``.config``.
        """
        replica, instance_id, net_type, c = self._items[idx]
        g = DCBAHeteroData.from_replica_record(replica, instance_id, net_type)
        g = CommunityToSize()(g)
        g.config = c
        return g
