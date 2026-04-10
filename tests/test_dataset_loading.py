"""Smoke tests for loading DCBA datasets via dcba-data-set."""

import pytest
from dcba_data_set.graph_io import load_report

from dcba.utils.paths import TEST_ABCD_REPORT, TEST_MABCD_REPORT


@pytest.mark.parametrize(
    "report_path",
    [TEST_ABCD_REPORT, TEST_MABCD_REPORT],
    ids=["abcd", "mabcd"],
)
class TestTestDatasetLoading:
    """Verify that the small test datasets load correctly."""

    def test_report_exists(self, report_path) -> None:
        """The report.json file is reachable at the expected path."""
        assert report_path.exists(), (
            f"report.json not found at {report_path}. "
            "Set DCBA_DATA_ROOT to the dcba-data-set data directory."
        )

    def test_load_returns_nonempty_configs(self, report_path) -> None:
        """load_report returns at least one config entry."""
        configs, _ = load_report(report_path)
        assert len(configs) > 0

    def test_load_returns_nonempty_graphs(self, report_path) -> None:
        """load_report returns at least one graph."""
        _, graphs = load_report(report_path)
        assert len(graphs) > 0
