"""Build a test DataLoader for evaluation: either a run's real test split, or a whole dataset."""

from pathlib import Path

from dcba_data_set.graph_io import load_dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import BaseTransform

from dcba.datamodule import ABCDDataModule
from dcba.dataset import ABCDBaseConfigScaler, ABCDDataset


def build_test_dataloader(
    dataset_root: Path,
    data_cfg: dict,
    scaler: ABCDBaseConfigScaler | None,
    transform: BaseTransform,
    node_transform: BaseTransform,
    *,
    whole_dataset_as_test: bool,
    seed: int,
) -> PyGDataLoader:
    """
    Build a test DataLoader over ``dataset_root``.

    :param dataset_root: Dataset directory (flat or chunked layout).
    :param data_cfg: The run's ``data`` config section, for ``batch_size``/``num_workers`` (and
        ``val_ratio``/``test_ratio`` when not ``whole_dataset_as_test``).
    :param scaler: Scaler applied to config features, matching what the checkpoint was trained
        with.
    :param transform: Config transform, matching what the checkpoint was trained with.
    :param node_transform: Node-feature transform, matching what the checkpoint was trained with.
    :param whole_dataset_as_test: When ``True``, every instance in ``dataset_root`` is used as the
        test set with no split -- for a dataset that exists solely as a held-out evaluation set
        (e.g. ``abcd-borderline``), where :class:`~dcba.datamodule.ABCDDataModule`'s split logic
        (which always requires a non-empty train partition) does not apply. When ``False``, builds
        an :class:`~dcba.datamodule.ABCDDataModule` and returns its own test split, reconstructed
        from ``data_cfg``/``seed`` exactly as during training.
    :param seed: RNG seed for the train/val/test split; unused when ``whole_dataset_as_test``.

    :returns: A PyG ``DataLoader`` over the resulting test set.
    """
    if whole_dataset_as_test:
        dataset = ABCDDataset(
            records=load_dataset(dataset_root),
            scaler=scaler,
            transform=transform,
            node_transform=node_transform,
            single_replica_per_instance=False,
        )
        return PyGDataLoader(
            dataset,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
        )

    datamodule = ABCDDataModule(
        dataset_root=dataset_root,
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        single_replica_per_instance=False,
        seed=seed,
        scaler=scaler,
        transform=transform,
        node_transform=node_transform,
    )
    datamodule.setup()
    return datamodule.test_dataloader()
