"""Unit tests for the pure reshaping/formatting logic in scripts/analyse_predictions.py."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from analyse_predictions import (  # noqa: E402
    _compute_metrics_df,
    _load_local_predictions_table,
    _reshape_predictions,
    _source_labels,
)


class TestReshapePredictions:
    """Tests for _reshape_predictions against the supcon wrapper's three-kind layout."""

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
        _, _, keys, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        assert keys == ["n", "xi"]

    def test_predictions_has_one_entry_per_non_orig_kind(self) -> None:
        """Predictions has one array per kind suffix other than 'orig'."""
        _, predictions, _, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        assert set(predictions) == {"regr", "crsm"}

    def test_arrays_are_aligned_by_sample_and_kind(self) -> None:
        """orig/regr/crsm rows land in the array row matching their sample index."""
        orig, predictions, _, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        np.testing.assert_array_equal(orig, [[100.0, 0.3], [200.0, 0.5]])
        np.testing.assert_array_equal(predictions["regr"], [[101.0, 0.31], [199.0, 0.49]])
        np.testing.assert_array_equal(predictions["crsm"], [[98.0, 0.29], [205.0, 0.52]])

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
        _, _, keys, _ = _reshape_predictions(columns, data)
        assert keys == ["n"]

    def test_meta_indexed_by_row_position_matching_arrays(self) -> None:
        """Meta holds one (instance, replica) row per sample, in the same order as the arrays."""
        _, _, _, meta = _reshape_predictions(self._COLUMNS, self._DATA)
        assert list(meta["instance"]) == ["inst-a", "inst-b"]
        assert list(meta["replica"]) == [0, 1]


class TestReshapePredictionsBaselineFormat:
    """Tests for _reshape_predictions against the baseline wrapper's two-kind layout."""

    _COLUMNS = ["sample", "instance", "replica", "n_norm", "xi_norm"]
    _DATA = [
        ["0-orig", "inst-a", 0, 100.0, 0.3],
        ["0-pred", "inst-a", 0, 101.0, 0.31],
        ["1-orig", "inst-b", 1, 200.0, 0.5],
        ["1-pred", "inst-b", 1, 199.0, 0.49],
    ]

    def test_predictions_has_a_single_pred_entry(self) -> None:
        """Only one non-orig kind ('pred') is present, so predictions has a single entry."""
        _, predictions, _, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        assert set(predictions) == {"pred"}

    def test_arrays_are_aligned_by_sample(self) -> None:
        """orig/pred rows land in the array row matching their sample index."""
        orig, predictions, _, _ = _reshape_predictions(self._COLUMNS, self._DATA)
        np.testing.assert_array_equal(orig, [[100.0, 0.3], [200.0, 0.5]])
        np.testing.assert_array_equal(predictions["pred"], [[101.0, 0.31], [199.0, 0.49]])


class TestComputeMetricsRows:
    """Tests for _compute_metrics_df."""

    # Index 0 ("n") is integer-valued per ABCD_INT_FEATURE_INDICES; index 1 ("xi") is not.
    _KEYS = ["n", "xi"]
    _ORIG = np.array([[10.0, 0.3], [20.0, 0.5], [15.0, 0.4]])
    _REGR = np.array([[10.0, 0.3], [20.0, 0.5], [15.0, 0.4]])
    _CROSS = np.array([[11.0, 0.31], [19.0, 0.49], [16.0, 0.42]])

    def test_one_row_per_variable_and_path(self) -> None:
        """The rows list has len(keys) * len(predictions) entries, one per (variable, mode) pair."""
        df = _compute_metrics_df(
            self._ORIG, {"regr": self._REGR, "crsm": self._CROSS}, self._KEYS, within_k=1
        )
        assert len(df) == 4
        assert set(df["variable"]) == {"n", "xi"}
        assert set(df["mode"]) == {"regr", "crsm"}

    def test_missing_within_k_accuracy_becomes_none(self) -> None:
        """Variables outside ABCD_INT_FEATURE_INDICES get None for within_k_accuracy."""
        df = _compute_metrics_df(
            self._ORIG, {"regr": self._REGR, "crsm": self._CROSS}, self._KEYS, within_k=1
        )
        assert df.loc[df["variable"] == "xi", "within_k_accuracy"].isna().all()
        assert df.loc[df["variable"] == "n", "within_k_accuracy"].notna().all()

    def test_single_prediction_path(self) -> None:
        """A single-mode predictions dict (baseline layout) produces one row per variable."""
        df = _compute_metrics_df(self._ORIG, {"pred": self._REGR}, self._KEYS, within_k=1)
        assert len(df) == 2
        assert set(df["mode"]) == {"pred"}


class TestLoadLocalPredictionsTable:
    """Tests for _load_local_predictions_table."""

    def test_round_trips_columns_and_data(self, tmp_path: Path) -> None:
        """A file written by compute_test_predictions.py reshapes identically to a wandb table."""
        columns = ["sample", "instance", "replica", "n_norm", "xi_norm"]
        data = [
            ["0-orig", "inst-a", 0, 100.0, 0.3],
            ["0-regr", "inst-a", 0, 101.0, 0.31],
            ["0-crsm", "inst-a", 0, 98.0, 0.29],
        ]
        path = tmp_path / "predictions_abcd-big.table.json"
        path.write_text(json.dumps({"columns": columns, "data": data}), encoding="utf-8")

        loaded_columns, loaded_data, _ = _load_local_predictions_table(path)

        assert loaded_columns == columns
        assert loaded_data == data
        orig, predictions, keys, _ = _reshape_predictions(loaded_columns, loaded_data)
        assert keys == ["n", "xi"]
        np.testing.assert_array_equal(orig, [[100.0, 0.3]])
        np.testing.assert_array_equal(predictions["regr"], [[101.0, 0.31]])
        np.testing.assert_array_equal(predictions["crsm"], [[98.0, 0.29]])

    def test_returns_meta_block(self, tmp_path: Path) -> None:
        """The meta block naming the run and dataset is returned as stored."""
        meta = {"run_id": "9dbudg6f", "dataset": "abcd-borderline", "test_scope": "whole_dataset"}
        path = tmp_path / "predictions_abcd-borderline.table.json"
        path.write_text(
            json.dumps({"columns": ["sample"], "data": [], "meta": meta}), encoding="utf-8"
        )

        assert _load_local_predictions_table(path)[2] == meta

    def test_meta_defaults_to_empty_for_older_files(self, tmp_path: Path) -> None:
        """A predictions file written before the meta block existed still loads."""
        path = tmp_path / "predictions_abcd-big.table.json"
        path.write_text(json.dumps({"columns": ["sample"], "data": []}), encoding="utf-8")

        assert _load_local_predictions_table(path)[2] == {}


class TestSourceLabels:
    """Tests for _source_labels."""

    def test_names_run_dataset_and_scope(self) -> None:
        """A full meta block yields a title and slug naming both the run and the dataset."""
        meta = {"run_id": "9dbudg6f", "dataset": "abcd-borderline", "test_scope": "whole_dataset"}

        title, slug = _source_labels(meta, fallback="ignored")

        assert title == "9dbudg6f -- abcd-borderline (whole dataset)"
        assert slug == "9dbudg6f-abcd-borderline"

    def test_spells_out_own_test_split_scope(self) -> None:
        """The run's own held-out split is labelled as such, not as the whole dataset."""
        meta = {"run_id": "9dbudg6f", "dataset": "abcd-big", "test_scope": "own_test_split"}

        title, slug = _source_labels(meta, fallback="ignored")

        assert title == "9dbudg6f -- abcd-big (own test split)"
        assert slug == "9dbudg6f-abcd-big"

    def test_omits_unknown_scope(self) -> None:
        """An absent or unrecognised scope drops the parenthetical instead of inventing one."""
        meta = {"run_id": "9dbudg6f", "dataset": "abcd-big"}

        assert _source_labels(meta, fallback="ignored")[0] == "9dbudg6f -- abcd-big"

    def test_falls_back_when_dataset_unknown(self) -> None:
        """Without a dataset there is nothing to append, so the fallback identifier is used."""
        assert _source_labels({}, fallback="predictions_abcd-big.table") == (
            "predictions_abcd-big.table",
            "predictions_abcd-big.table",
        )
