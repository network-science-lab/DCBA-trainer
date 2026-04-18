"""Generic dataset for DCBA configuration records."""

from pathlib import Path

from dcba_data_set.graph_io import load_report
from dcba_data_set.graph_io.data_models import ConfigRecord, DCBAHeteroData
from torch import Tensor, zeros
from torch.utils.data import Dataset

from dcba.dataset.transforms import ABCDConfigScaler, ABCDConfigToTensor


class ABCDDataset(Dataset):
    """
    Dataset of configuration tensors & Graphs for autoencoders training.

    Each item is a ``DCBAHeteroData`` containing ABCD configuration and generated from it
    graph — the input and reconstruction target for the autoencoder.

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
        graphs: dict[str, DCBAHeteroData],
        scaler: ABCDConfigScaler | None = None,
        unique_configs: bool = True,
    ) -> None:
        """Initialise the dataset, eagerly converting records to tensors and applying the scaler."""
        super().__init__()
        to_tensor = ABCDConfigToTensor()
        self._graphs = []
        _configs = []
        for key in configs.keys():
            if unique_configs:
                _configs.append(configs[key])
                # Single graph for config_autoencoder to keep consistency of the data flow
                # architecture. This model will remove this object because it is unnecessary
                # for its training
                self._graphs.append(graphs[key][0])
            else:
                _configs.extend([configs[key]] * len(graphs[key]))
                self._graphs.extend(graphs[key])
        self._tensors: list[Tensor] = [
            scaler(to_tensor(r)) if scaler is not None else to_tensor(r) for r in _configs
        ]
        assert len(self._graphs) == len(self._tensors)

    @classmethod
    def from_report(
        cls,
        report_path: Path,
        scaler: ABCDConfigScaler | None = None,
    ) -> "ABCDDataset":
        """
        Build a dataset by loading a report.json manifest.

        :param report_path: Path to the ``report.json`` produced by
            :class:`~dcba_data_set.ds_generator.DatasetGenerator`.
        :param scaler: Optional scaler applied after the tensor transform.

        :returns: A :class:`ConfigDataset` constructed from all configs in the report.
        """
        configs, graphs = load_report(report_path)
        return cls(configs=configs, graphs=graphs, scaler=scaler)

    def __len__(self) -> int:
        """Return the number of config instances in the dataset."""
        return len(self._tensors)

    def __getitem__(self, idx: int) -> DCBAHeteroData:
        """
        Return the config tensor at the given index as an (input, target) pair.

        :param idx: Index of the config instance.

        :returns: A DCBAHeteroData with config of independent float tensors of shape ``(9,)``.
        """
        t = self._tensors[idx]
        g = self._graphs[idx].clone()
        # TODO: store data paths and load them here
        # TODO: move it to DCBAHeteroData.from_abcd_files / from_mabcd_files
        # TODO: __inc__ & __cat_dim__ for custom attributes
        g["actor"].x = zeros((len(g.actors_map), 5))
        g["actor"].config = t
        g["actor"].y = t.clone()
        g.actors_map = None
        g.layers_map = None
        return g
