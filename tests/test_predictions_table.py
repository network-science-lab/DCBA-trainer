"""Unit tests for the shared orig/regr/crsm predictions-table schema."""

from dcba.eval.predictions_table import build_predictions_table

_KEYS = ["n", "xi"]
_ROWS: list[tuple[str, int, list[float], list[float], list[float]]] = [
    ("inst-a", 0, [100.0, 0.3], [101.0, 0.31], [98.0, 0.29]),
    ("inst-b", 2, [200.0, 0.4], [199.0, 0.41], [205.0, 0.38]),
]


class TestBuildPredictionsTable:
    """Tests for build_predictions_table."""

    def test_emits_three_rows_per_sample_in_order(self) -> None:
        """Each sample yields its orig, regr, and crsm row, in that order and index-tagged."""
        _, data = build_predictions_table(_ROWS, _KEYS, has_scaler=True)

        assert len(data) == 3 * len(_ROWS)
        assert [row[0] for row in data] == [
            "0-orig",
            "0-regr",
            "0-crsm",
            "1-orig",
            "1-regr",
            "1-crsm",
        ]
        assert data[0] == ["0-orig", "inst-a", 0, 100.0, 0.3]
        assert data[1] == ["0-regr", "inst-a", 0, 101.0, 0.31]
        assert data[2] == ["0-crsm", "inst-a", 0, 98.0, 0.29]
        assert data[3] == ["1-orig", "inst-b", 2, 200.0, 0.4]

    def test_columns_unsuffixed_with_scaler(self) -> None:
        """With a scaler the values are in raw ABCD units, so column names carry no suffix."""
        columns, _ = build_predictions_table(_ROWS, _KEYS, has_scaler=True)

        assert columns == ["sample", "instance", "replica", "n", "xi"]

    def test_columns_suffixed_without_scaler(self) -> None:
        """Without a scaler the values stay normalised, which the ``_norm`` suffix flags."""
        columns, _ = build_predictions_table(_ROWS, _KEYS, has_scaler=False)

        assert columns == ["sample", "instance", "replica", "n_norm", "xi_norm"]

    def test_handles_no_rows(self) -> None:
        """An empty test set still produces a well-formed, empty table."""
        columns, data = build_predictions_table([], _KEYS, has_scaler=True)

        assert columns == ["sample", "instance", "replica", "n", "xi"]
        assert data == []
