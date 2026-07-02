"""Per-variable regression diagnostics for ABCD config predictions.

Note: these metrics assume a fixed-length ABCD parameter vector; mABCD has not been
tackled yet.
"""

import numpy as np


def per_variable_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    keys: list[str],
    int_indices: list[int],
    within_k: int,
) -> dict[str, dict[str, float]]:
    """
    Compute per-variable regression diagnostics comparing predicted and true config vectors.

    :param y_true: ``(N, len(keys))`` ground-truth values.
    :param y_pred: ``(N, len(keys))`` predicted values, same shape as ``y_true``.
    :param keys: Name of each variable, in column order.
    :param int_indices: Indices (into ``keys``) of integer-valued variables; these
        additionally get a ``within_k_accuracy`` entry.
    :param within_k: Tolerance for ``within_k_accuracy`` -- a prediction counts as correct
        when its rounded value is within this many units of the true integer value.

    :returns: Dict mapping each variable name to a dict of metric name -> value. Metrics are
        ``r2``, ``pearson_r``, ``relative_error``, ``nrmse``, and (for ``int_indices`` columns
        only) ``within_k_accuracy``. ``r2``/``nrmse`` are reported as ``nan`` rather than
        raising when ``y_true`` is constant (zero variance).
    """
    results: dict[str, dict[str, float]] = {}
    for i, key in enumerate(keys):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        results[key] = _column_metrics(true_col, pred_col, is_integer=i in int_indices)
        if i in int_indices:
            results[key]["within_k_accuracy"] = _within_k_accuracy(true_col, pred_col, within_k)
    return results


def _column_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_integer: bool) -> dict[str, float]:
    """
    Compute r2, pearson_r, relative_error, and nrmse for a single variable column.

    :param y_true: ``(N,)`` ground-truth values for one variable.
    :param y_pred: ``(N,)`` predicted values for one variable.
    :param is_integer: Whether this variable is integer/count-valued; passed through to
        :func:`_mean_relative_error` to pick an appropriate denominator floor.

    :returns: Dict with keys ``r2``, ``pearson_r``, ``relative_error``, ``nrmse``.
    """
    residuals = y_pred - y_true
    true_std = float(np.std(y_true))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson_r": _pearson_r(y_true, y_pred),
        "relative_error": _mean_relative_error(y_true, y_pred, is_integer),
        "nrmse": rmse / true_std if true_std > 0 else float("nan"),
    }


def _mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray, is_integer: bool) -> float:
    """
    Compute the mean absolute percentage error: the per-sample relative error, then averaged.

    Some ABCD parameters (e.g. ``xi``, ``nout``) can be exactly 0 for a given sample, so the
    denominator is floored rather than using the raw ``|y_true|``, to avoid a division by zero.
    For integer/count variables (e.g. ``nout``), which commonly take the value 0 exactly, the
    floor is 1 -- a unit of error against a true value of 0 then reads as 100%. For continuous
    variables, which essentially never land on exactly 0, the floor is machine epsilon --
    following the same convention as ``sklearn.metrics.mean_absolute_percentage_error``.

    :param y_true: ``(N,)`` ground-truth values for one variable.
    :param y_pred: ``(N,)`` predicted values for one variable.
    :param is_integer: Whether this variable is integer/count-valued.

    :returns: Mean of ``|pred_i - true_i| / max(|true_i|, floor)`` over all samples.
    """
    floor = 1.0 if is_integer else np.finfo(np.float64).eps
    denom = np.maximum(np.abs(y_true), floor)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient between two 1-D arrays.

    :param y_true: ``(N,)`` ground-truth values.
    :param y_pred: ``(N,)`` predicted values.

    :returns: Pearson r in ``[-1, 1]``, or ``nan`` if either input has zero variance.
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _within_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Compute the fraction of predictions whose rounded value is within k of the true value.

    :param y_true: ``(N,)`` ground-truth integer-valued values.
    :param y_pred: ``(N,)`` predicted values (rounded before comparison).
    :param k: Tolerance -- a prediction counts as correct when ``abs(round(pred) - true) <= k``.

    :returns: Fraction of samples in ``[0, 1]`` satisfying the tolerance.
    """
    return float(np.mean(np.abs(np.round(y_pred) - y_true) <= k))
