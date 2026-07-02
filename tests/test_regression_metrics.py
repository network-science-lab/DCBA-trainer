"""Unit tests for dcba.training.metrics.regression."""

import numpy as np

from dcba.training.metrics import per_variable_regression_metrics


class TestPerVariableRegressionMetrics:
    """Tests for per_variable_regression_metrics."""

    def test_perfect_predictions_are_ideal(self) -> None:
        """r2, pearson_r, relative_error, and nrmse hit their ideal values when pred == true."""
        rng = np.random.default_rng(0)
        y_true = rng.normal(size=(20, 2))
        y_pred = y_true.copy()

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["a", "b"], int_indices=[], within_k=0
        )

        for key in ("a", "b"):
            assert metrics[key]["r2"] == 1.0
            assert metrics[key]["pearson_r"] == 1.0
            assert metrics[key]["relative_error"] == 0.0
            assert metrics[key]["nrmse"] == 0.0

    def test_noisy_predictions_land_in_expected_range(self) -> None:
        """Small Gaussian noise around the true values yields a high but imperfect r2/pearson_r."""
        rng = np.random.default_rng(1)
        y_true = rng.normal(loc=10.0, scale=5.0, size=(200, 1))
        y_pred = y_true + rng.normal(scale=0.5, size=y_true.shape)

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["x"], int_indices=[], within_k=0
        )["x"]

        assert 0.9 < metrics["r2"] < 1.0
        assert 0.9 < metrics["pearson_r"] < 1.0
        assert 0.0 < metrics["relative_error"] < 0.2
        assert 0.0 < metrics["nrmse"] < 0.2

    def test_relative_error_is_mean_of_per_sample_ratios(self) -> None:
        """relative_error averages |pred_i - true_i| / |true_i| per sample, not MAE / mean(true)."""
        y_true = np.array([[1.0], [10.0]])
        y_pred = np.array([[1.5], [11.0]])
        # Per-sample ratios: |0.5|/1 = 0.5, |1.0|/10 = 0.1 -> mean = 0.3.
        # Ratio of aggregates would instead give MAE / mean(|true|) = 0.75 / 5.5 ~= 0.136.

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["x"], int_indices=[], within_k=0
        )["x"]

        assert metrics["relative_error"] == 0.3

    def test_within_k_accuracy_matches_hand_computed_value(self) -> None:
        """within_k_accuracy for k=1 matches a manually verified fraction on a fixed array."""
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        # Rounded predictions: 1, 3, 3, 10 -- within 1 of true for rows 0, 1, 2 but not row 3.
        y_pred = np.array([[1.1], [2.6], [2.9], [10.0]])

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["n"], int_indices=[0], within_k=1
        )["n"]

        assert metrics["within_k_accuracy"] == 0.75

    def test_non_integer_variable_has_no_within_k_key(self) -> None:
        """Variables outside int_indices do not get a within_k_accuracy entry."""
        y_true = np.array([[0.1], [0.2], [0.3]])
        y_pred = np.array([[0.1], [0.2], [0.3]])

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["xi"], int_indices=[], within_k=1
        )["xi"]

        assert "within_k_accuracy" not in metrics

    def test_constant_true_column_returns_nan_without_raising(self) -> None:
        """A constant y_true column yields nan for r2/relative_error/nrmse, not an exception."""
        y_true = np.full((10, 1), 5.0)
        y_pred = np.linspace(4.0, 6.0, 10).reshape(-1, 1)

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["const"], int_indices=[], within_k=1
        )["const"]

        assert np.isnan(metrics["r2"])
        assert np.isnan(metrics["nrmse"])
        # relative_error is a per-sample ratio, well-defined even when y_true is constant.
        assert not np.isnan(metrics["relative_error"])

    def test_zero_true_values_floor_relative_error_instead_of_nan(self) -> None:
        """A y_true of all zeros yields a large but finite relative_error, never nan/inf."""
        y_true = np.zeros((10, 1))
        y_pred = np.linspace(-1.0, 1.0, 10).reshape(-1, 1)

        metrics = per_variable_regression_metrics(
            y_true, y_pred, keys=["zero"], int_indices=[], within_k=1
        )["zero"]

        assert np.isfinite(metrics["relative_error"])
        assert metrics["relative_error"] > 1e6
        assert np.isnan(metrics["r2"])
        assert np.isnan(metrics["nrmse"])
