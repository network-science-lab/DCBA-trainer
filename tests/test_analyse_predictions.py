"""Unit tests for the pure reshaping/formatting logic in scripts/analyse_predictions.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from analyse_predictions import _build_metrics_table, _reshape_predictions  # noqa: E402


class TestReshapePredictions:
    """Tests for _reshape_predictions."""

    _COLUMNS = ["sample", "instance", "replica", "n_norm", "xi_norm"]
    _DATA = [
        ["0-orig", "inst-a", 0, 100.0, 0.3],
        ["0-regr", "inst-a", 0, 101.0, 0.31],
        ["0-crsm", "inst-a", 0, 98.0, 0.29],
        ["1-orig", "inst-b", 1, 200.0, 0.5],
        ["1-regr", "inst-b", 1, 199.0, 0.49],
        ["1-crsm", "inst-b", 1, 205.0, 0.52],
    ]

    def test_strips_norm_suffix_from_keys(self) -> None:
        """Variable names have the _norm suffix stripped when no scaler was used."""
        _, _, _, keys = _reshape_predictions(self._COLUMNS, self._DATA)
        assert keys == ["n", "xi"]

    def test_arrays_are_aligned_by_sample_and_kind(self) -> None:
        """orig/regr/cross rows land in the array row matching their sample index."""
        orig, regr, cross, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        np.testing.assert_array_equal(orig, [[100.0, 0.3], [200.0, 0.5]])
        np.testing.assert_array_equal(regr, [[101.0, 0.31], [199.0, 0.49]])
        np.testing.assert_array_equal(cross, [[98.0, 0.29], [205.0, 0.52]])

    def test_sample_ids_sorted_numerically_not_lexically(self) -> None:
        """Sample ids like '2' and '10' sort as 2 < 10, not lexically ('10' < '2')."""
        columns = ["sample", "instance", "replica", "n_norm"]
        data = [
            ["2-orig", "inst-a", 0, 3.0],
            ["2-regr", "inst-a", 0, 3.1],
            ["2-crsm", "inst-a", 0, 2.9],
            ["10-orig", "inst-b", 1, 4.0],
            ["10-regr", "inst-b", 1, 4.1],
            ["10-crsm", "inst-b", 1, 3.9],
        ]
        orig, _, _, _ = _reshape_predictions(columns, data)
        np.testing.assert_array_equal(orig, [[3.0], [4.0]])

    def test_keys_without_scaler_suffix_are_unchanged(self) -> None:
        """Column names without a _norm suffix (scaler present) pass through unchanged."""
        columns = ["sample", "instance", "replica", "n"]
        data = [
            ["0-orig", "inst-a", 0, 100.0],
            ["0-regr", "inst-a", 0, 101.0],
            ["0-crsm", "inst-a", 0, 98.0],
        ]
        _, _, _, keys = _reshape_predictions(columns, data)
        assert keys == ["n"]


class TestBuildMetricsTable:
    """Tests for _build_metrics_table."""

    def test_one_row_per_variable_and_path(self) -> None:
        """The table has len(keys) * len(per_path_metrics) rows."""
        per_path_metrics = {
            "regr": {"n": {"r2": 1.0, "pearson_r": 1.0, "relative_error": 0.0, "nrmse": 0.0}},
            "cross": {"n": {"r2": 0.9, "pearson_r": 0.95, "relative_error": 0.1, "nrmse": 0.2}},
        }
        table = _build_metrics_table(per_path_metrics, keys=["n"])
        assert len(table.data) == 2
        assert {row[0] for row in table.data} == {"n"}
        assert {row[1] for row in table.data} == {"regr", "cross"}

    def test_missing_within_k_accuracy_becomes_none(self) -> None:
        """Non-integer variables without within_k_accuracy get None in that column."""
        per_path_metrics = {
            "regr": {"xi": {"r2": 1.0, "pearson_r": 1.0, "relative_error": 0.0, "nrmse": 0.0}},
        }
        table = _build_metrics_table(per_path_metrics, keys=["xi"])
        assert table.data[0][-1] is None
