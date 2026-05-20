"""Smoke tests for loading DCBA datasets via dcba-data-set."""

import pytest
from dcba_data_set.graph_io import load_dataset

from dcba.utils.paths import TEST_ABCD_DATASET, TEST_MABCD_DATASET


@pytest.mark.parametrize(
    "dataset_root",
    [TEST_ABCD_DATASET, TEST_MABCD_DATASET],
    ids=["abcd", "mabcd"],
)
class TestTestDatasetLoading:
    """Verify that the small test datasets load correctly."""

    def test_dataset_exists(self, dataset_root) -> None:
        """The dataset directory is reachable at the expected path."""
        assert dataset_root.exists(), (
            f"Dataset directory not found at {dataset_root}. "
            "Set DCBA_DATA_ROOT to the dcba-data-set data directory."
        )

    def test_load_returns_nonempty_records(self, dataset_root) -> None:
        """load_dataset returns at least one InstanceRecord."""
        records = load_dataset(dataset_root)
        assert len(records) > 0

    def test_load_records_have_replicas(self, dataset_root) -> None:
        """Each InstanceRecord contains at least one replica."""
        records = load_dataset(dataset_root)
        assert any(len(r.replicas) > 0 for r in records)
