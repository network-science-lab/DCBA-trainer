"""Per-variable regression diagnostics for a supcon test run, re-uploaded to wandb.

Fetches the ``test/predictions`` wandb.Table logged by
:meth:`~dcba.wrappers.supcon.DCBASupConWrapper.on_test_epoch_end`, computes per-variable
regression metrics for both the ``regr`` (config-encoder self-reconstruction) and ``cross``
(graph-encoder -> theta) prediction paths, and logs a metrics table plus per-variable
scatter/residual plots back to the same run under the ``analysis/`` prefix.

Usage::

    uv run python scripts/analyse_predictions.py <entity>/<project>/<run_id>
"""

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from wandb.apis.public import Run

from dcba.dataset import ABCD_INT_FEATURE_INDICES
from dcba.training.metrics import per_variable_regression_metrics
from dcba.training.metrics.plotting import residual_histogram, scatter_pred_vs_true

_METRIC_COLUMNS = [
    "variable",
    "path",
    "r2",
    "pearson_r",
    "relative_error",
    "nrmse",
    "within_k_accuracy",
]


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", help="wandb run path, e.g. entity/project/run_id")
    parser.add_argument(
        "--within-k",
        type=int,
        default=1,
        help="Tolerance for the within-k accuracy metric on integer-valued ABCD parameters.",
    )
    return parser.parse_args()


def _fetch_predictions_table(run: Run) -> tuple[list[str], list[list]]:
    """
    Download the ``test/predictions`` wandb.Table logged during testing for a run.

    :param run: The wandb run to fetch the table from.

    :returns: Tuple of ``(columns, data)`` as logged in the table's underlying JSON file.
    """
    history_rows = list(run.scan_history(keys=["test/predictions"]))
    if not history_rows:
        raise ValueError(
            f"Run '{'/'.join(run.path)}' has no 'test/predictions' entries in its history."
        )
    table_ref = history_rows[-1]["test/predictions"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        run.file(table_ref["path"]).download(root=tmp_dir, replace=True)
        table_path = Path(tmp_dir) / table_ref["path"]
        with table_path.open(encoding="utf-8") as f:
            table_json = json.load(f)

    return table_json["columns"], table_json["data"]


def _reshape_predictions(
    columns: list[str], data: list[list]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Reshape the flat ``{i}-orig`` / ``{i}-regr`` / ``{i}-crsm`` table rows into arrays.

    :param columns: Table column names, as produced by
        :meth:`~dcba.wrappers.supcon.DCBASupConWrapper.on_test_epoch_end` --
        ``["sample", "instance", "replica", <one column per ABCD variable>]``.
    :param data: Table rows in the same column order.

    :returns: Tuple ``(orig, regr, cross, keys)`` where the first three are ``(N, 9)`` arrays
        aligned by sample, and ``keys`` are the variable names with any ``_norm`` suffix
        stripped.
    """
    variable_columns = [c for c in columns if c not in ("sample", "instance", "replica")]
    keys = [c[: -len("_norm")] if c.endswith("_norm") else c for c in variable_columns]
    var_start = len(columns) - len(variable_columns)

    rows_by_kind: dict[str, dict[str, list[float]]] = {"orig": {}, "regr": {}, "crsm": {}}
    for row in data:
        sample_id, kind = row[0].rsplit("-", 1)
        rows_by_kind[kind][sample_id] = row[var_start:]

    sample_ids = sorted(rows_by_kind["orig"], key=int)
    orig = np.array([rows_by_kind["orig"][i] for i in sample_ids], dtype=float)
    regr = np.array([rows_by_kind["regr"][i] for i in sample_ids], dtype=float)
    cross = np.array([rows_by_kind["crsm"][i] for i in sample_ids], dtype=float)
    return orig, regr, cross, keys


def _build_metrics_table(
    per_path_metrics: dict[str, dict[str, dict[str, float]]], keys: list[str]
) -> wandb.Table:
    """
    Flatten per-path, per-variable metric dicts into a single wandb.Table.

    :param per_path_metrics: Maps path name (``"regr"``/``"cross"``) to the dict returned by
        :func:`~dcba.training.metrics.per_variable_regression_metrics`.
    :param keys: Variable names, in column order.

    :returns: One row per ``(variable, path)`` pair, columns as in :data:`_METRIC_COLUMNS`.
    """
    rows = []
    for path_name, metrics in per_path_metrics.items():
        for key in keys:
            m = metrics[key]
            rows.append(
                [
                    key,
                    path_name,
                    m["r2"],
                    m["pearson_r"],
                    m["relative_error"],
                    m["nrmse"],
                    m.get("within_k_accuracy"),
                ]
            )
    return wandb.Table(columns=_METRIC_COLUMNS, data=rows)


def _print_metrics_table(table: wandb.Table) -> None:
    """Print a wandb.Table's rows as a plain-text table on stdout."""
    header = " | ".join(_METRIC_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in table.data:
        print(
            " | ".join(
                "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v) for v in row
            )
        )


def _build_analysis_payload(
    orig: np.ndarray, regr: np.ndarray, cross: np.ndarray, keys: list[str], within_k: int
) -> dict[str, wandb.Table | wandb.Image]:
    """
    Compute per-variable metrics and plots for both prediction paths.

    :param orig: ``(N, 9)`` ground-truth config values.
    :param regr: ``(N, 9)`` config-encoder self-reconstruction predictions.
    :param cross: ``(N, 9)`` graph-encoder -> theta cross-modal predictions.
    :param keys: Variable names, in column order.
    :param within_k: Tolerance for the within-k accuracy metric.

    :returns: Dict ready to pass to ``wandb.Run.log``, keyed under the ``analysis/`` prefix.
    """
    payload: dict[str, wandb.Table | wandb.Image] = {}
    per_path_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for path_name, predictions in (("regr", regr), ("cross", cross)):
        per_path_metrics[path_name] = per_variable_regression_metrics(
            orig, predictions, keys, ABCD_INT_FEATURE_INDICES, within_k
        )
        for i, key in enumerate(keys):
            scatter_fig = scatter_pred_vs_true(orig[:, i], predictions[:, i], key)
            payload[f"analysis/{path_name}/{key}/scatter"] = wandb.Image(scatter_fig)
            plt.close(scatter_fig)

            residual_fig = residual_histogram(orig[:, i], predictions[:, i], key)
            payload[f"analysis/{path_name}/{key}/residuals"] = wandb.Image(residual_fig)
            plt.close(residual_fig)

    payload["analysis/regression_metrics"] = _build_metrics_table(per_path_metrics, keys)
    return payload


def main() -> None:
    """Fetch predictions for a run, compute diagnostics, and re-upload them to the same run."""
    args = _parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    columns, data = _fetch_predictions_table(run)
    orig, regr, cross, keys = _reshape_predictions(columns, data)

    payload = _build_analysis_payload(orig, regr, cross, keys, args.within_k)
    _print_metrics_table(payload["analysis/regression_metrics"])

    with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="must") as write_run:
        write_run.log(payload)


if __name__ == "__main__":
    main()
