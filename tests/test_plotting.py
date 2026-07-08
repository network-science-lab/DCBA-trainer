"""Unit tests for dcba.training.metrics.plotting."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from dcba.training.metrics.plotting import residual_histogram, scatter_pred_vs_true


class TestScatterPredVsTrue:
    """Tests for scatter_pred_vs_true."""

    def test_returns_a_figure(self) -> None:
        """The function returns a matplotlib Figure without raising."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        fig = scatter_pred_vs_true(y_true, y_pred, "n")

        assert isinstance(fig, Figure)
        plt.close(fig)


class TestResidualHistogram:
    """Tests for residual_histogram."""

    def test_returns_a_figure(self) -> None:
        """The function returns a matplotlib Figure without raising."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        fig = residual_histogram(y_true, y_pred, "n")

        assert isinstance(fig, Figure)
        plt.close(fig)
