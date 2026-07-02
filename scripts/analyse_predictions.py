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
import csv
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    parser.add_argument(
        "run_path",
        nargs="?",
        default="network-science-lab/DCBA/vi1csr7r",
        help="wandb run path, e.g. entity/project/run_id",
    )
    parser.add_argument(
        "--within-k",
        type=int,
        default=1,
        help="Tolerance for the within-k accuracy metric on integer-valued ABCD parameters.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        # default="./dump",
        help="If set, also write the analysis payload (plots + metrics table) to this local "
        "directory for offline inspection.",
    )
    return parser.parse_args()


def _fetch_predictions_table(run: Run) -> tuple[list[str], list[list]]:
    """
    Download the ``test/predictions`` wandb.Table logged during testing for a run.

    Reads the run-table artifact rather than the history-logged media file: the latter is
    silently truncated to ``wandb.Table.MAX_ROWS`` (10000) rows, while the artifact retains
    the full table.

    :param run: The wandb run to fetch the table from.

    :returns: Tuple of ``(columns, data)`` as logged in the table's underlying JSON file.
    """
    artifact = next((a for a in run.logged_artifacts() if a.type == "run_table"), None)
    if artifact is None:
        run_path = "/".join(run.path)
        raise ValueError(f"Run '{run_path}' has no 'test/predictions' run-table artifact.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = Path(artifact.download(root=tmp_dir))
        table_path = next(artifact_dir.rglob("*.table.json"))
        with table_path.open(encoding="utf-8") as f:
            table_json = json.load(f)

    return table_json["columns"], table_json["data"]


def _reshape_predictions(
    columns: list[str], data: list[list]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Reshape the flat ``{i}-orig`` / ``{i}-regr`` / ``{i}-crsm`` table rows into arrays.

    :param columns: Table column names, as produced by
        :meth:`~dcba.wrappers.supcon.DCBASupConWrapper.on_test_epoch_end` --
        ``["sample", "instance", "replica", <one column per ABCD variable>]``.
    :param data: Table rows in the same column order.

    :returns: Tuple ``(orig, regr, cross, keys, meta)`` where the first three are ``(N, 9)``
        arrays aligned by sample, ``keys`` are the variable names with any ``_norm`` suffix
        stripped, and ``meta`` is a DataFrame with ``instance``/``replica`` columns, indexed by
        the same row position as the arrays.
    """
    variable_columns = [c for c in columns if c not in ("sample", "instance", "replica")]
    keys = [c[: -len("_norm")] if c.endswith("_norm") else c for c in variable_columns]
    var_start = len(columns) - len(variable_columns)
    instance_idx = columns.index("instance")
    replica_idx = columns.index("replica")

    rows_by_kind: dict[str, dict[str, list[float]]] = {"orig": {}, "regr": {}, "crsm": {}}
    metadata: dict[str, tuple[str, int]] = {}
    for row in data:
        sample_id, kind = row[0].rsplit("-", 1)
        rows_by_kind[kind][sample_id] = row[var_start:]
        if kind == "orig":
            metadata[sample_id] = (row[instance_idx], row[replica_idx])

    sample_ids = sorted(rows_by_kind["orig"], key=int)
    orig = np.array([rows_by_kind["orig"][i] for i in sample_ids], dtype=float)
    regr = np.array([rows_by_kind["regr"][i] for i in sample_ids], dtype=float)
    cross = np.array([rows_by_kind["crsm"][i] for i in sample_ids], dtype=float)
    meta = pd.DataFrame([metadata[i] for i in sample_ids], columns=["instance", "replica"])
    return orig, regr, cross, keys, meta


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


def _dump_payload(payload: dict[str, wandb.Table | wandb.Image], dump_dir: Path) -> None:
    """
    Write a wandb log payload to local files for offline inspection.

    Each key becomes a path under ``dump_dir`` (slashes in the key become subdirectories):
    ``wandb.Image`` entries are saved as PNGs, ``wandb.Table`` entries as CSVs.

    :param payload: Mapping of wandb log key to a ``wandb.Image`` or ``wandb.Table`` value.
    :param dump_dir: Directory to write into; created (including subdirectories) if missing.
    """
    for key, value in payload.items():
        dest = dump_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, wandb.Image):
            value.image.save(dest.with_suffix(".png"))
        elif isinstance(value, wandb.Table):
            with dest.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(value.columns)
                writer.writerows(value.data)


def main() -> None:
    """Fetch predictions for a run, compute diagnostics, and re-upload them to the same run."""
    args = _parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    columns, data = _fetch_predictions_table(run)
    orig, regr, cross, keys, meta = _reshape_predictions(columns, data)

    payload = _build_analysis_payload(orig, regr, cross, keys, args.within_k)
    _print_metrics_table(payload["analysis/regression_metrics"])

    if args.dump_dir is not None:
        _dump_payload(payload, args.dump_dir)
        print(f"Dumped analysis payload to {args.dump_dir}")

    with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="must") as write_run:
        write_run.log(payload)


if __name__ == "__main__":
    main()
