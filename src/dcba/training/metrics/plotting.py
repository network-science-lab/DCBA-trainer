"""Per-variable diagnostic plots for ABCD config predictions."""

import matplotlib

# Headless backend: these figures are built for wandb upload on training servers with no
# display, never shown interactively. Must be set before importing pyplot.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def scatter_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, variable_name: str) -> Figure:
    """
    Plot predicted vs. true values for one variable, with a y = x reference line.

    :param y_true: ``(N,)`` ground-truth values for one variable.
    :param y_pred: ``(N,)`` predicted values for one variable.
    :param variable_name: Name of the variable, used for the axis labels and title.

    :returns: The created figure. Caller owns it and is responsible for closing it.
    """
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(f"true {variable_name}")
    ax.set_ylabel(f"predicted {variable_name}")
    ax.set_title(f"{variable_name}: predicted vs. true")
    fig.tight_layout()
    return fig


def residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, variable_name: str) -> Figure:
    """
    Plot a histogram of residuals (predicted - true) for one variable.

    :param y_true: ``(N,)`` ground-truth values for one variable.
    :param y_pred: ``(N,)`` predicted values for one variable.
    :param variable_name: Name of the variable, used for the axis labels and title.

    :returns: The created figure. Caller owns it and is responsible for closing it.
    """
    fig, ax = plt.subplots()
    residuals = y_pred - y_true
    ax.hist(residuals, bins=30, color="steelblue", edgecolor="black")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(f"residual ({variable_name}: predicted - true)")
    ax.set_ylabel("count")
    ax.set_title(f"{variable_name}: residuals")
    fig.tight_layout()
    return fig
