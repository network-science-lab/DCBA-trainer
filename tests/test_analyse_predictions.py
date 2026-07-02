"""Unit tests for the pure reshaping/formatting logic in scripts/analyse_predictions.py."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from analyse_predictions import _compute_metrics_rows, _reshape_predictions  # noqa: E402


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
        _, _, _, keys, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        assert keys == ["n", "xi"]

    def test_arrays_are_aligned_by_sample_and_kind(self) -> None:
        """orig/regr/cross rows land in the array row matching their sample index."""
        orig, regr, cross, _, _ = _reshape_predictions(self._COLUMNS, self._DATA)
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
        orig, _, _, _, _ = _reshape_predictions(columns, data)
        np.testing.assert_array_equal(orig, [[3.0], [4.0]])

    def test_keys_without_scaler_suffix_are_unchanged(self) -> None:
        """Column names without a _norm suffix (scaler present) pass through unchanged."""
        columns = ["sample", "instance", "replica", "n"]
        data = [
            ["0-orig", "inst-a", 0, 100.0],
            ["0-regr", "inst-a", 0, 101.0],
            ["0-crsm", "inst-a", 0, 98.0],
        ]
        _, _, _, keys, _ = _reshape_predictions(columns, data)
        assert keys == ["n"]

    def test_meta_indexed_by_row_position_matching_arrays(self) -> None:
        """Meta holds one (instance, replica) row per sample, in the same order as the arrays."""
        _, _, _, _, meta = _reshape_predictions(self._COLUMNS, self._DATA)
        assert list(meta["instance"]) == ["inst-a", "inst-b"]
        assert list(meta["replica"]) == [0, 1]


class TestComputeMetricsRows:
    """Tests for _compute_metrics_rows."""

    # Index 0 ("n") is integer-valued per ABCD_INT_FEATURE_INDICES; index 1 ("xi") is not.
    _KEYS = ["n", "xi"]
    _ORIG = np.array([[10.0, 0.3], [20.0, 0.5], [15.0, 0.4]])
    _REGR = np.array([[10.0, 0.3], [20.0, 0.5], [15.0, 0.4]])
    _CROSS = np.array([[11.0, 0.31], [19.0, 0.49], [16.0, 0.42]])

    def test_one_row_per_variable_and_path(self) -> None:
        """The rows list has len(keys) * len(paths) entries, one per (variable, path) pair."""
        rows = _compute_metrics_rows(self._ORIG, self._REGR, self._CROSS, self._KEYS, within_k=1)
        assert len(rows) == 4
        assert {row[0] for row in rows} == {"n", "xi"}
        assert {row[1] for row in rows} == {"regr", "cross"}

    def test_missing_within_k_accuracy_becomes_none(self) -> None:
        """Variables outside ABCD_INT_FEATURE_INDICES get None for within_k_accuracy."""
        rows = _compute_metrics_rows(self._ORIG, self._REGR, self._CROSS, self._KEYS, within_k=1)
        assert all(row[-1] is None for row in rows if row[0] == "xi")
        assert all(row[-1] is not None for row in rows if row[0] == "n")
