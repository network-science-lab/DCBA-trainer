"""Per-variable regression diagnostics for a supcon test run, re-uploaded to wandb."""

import argparse
import csv
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.backends.backend_pdf import PdfPages
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
        default="network-science-lab/DCBA/j7igczgu",
        help="wandb run path, e.g. entity/project/run_id",
    )
    parser.add_argument(
        "--within-k",
        type=int,
        default=10,
        help="Tolerance for the within-k accuracy metric on integer-valued ABCD parameters.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=None,
        help="If set, also write the analysis payload to this directory for offline inspection.",
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


def _compute_metrics_rows(
    orig: np.ndarray, regr: np.ndarray, cross: np.ndarray, keys: list[str], within_k: int
) -> list[list]:
    """
    Compute per-variable regression metrics for both prediction paths as flat table rows.

    :param orig: ``(N, 9)`` ground-truth config values.
    :param regr: ``(N, 9)`` config-encoder self-reconstruction predictions.
    :param cross: ``(N, 9)`` graph-encoder -> theta cross-modal predictions.
    :param keys: Variable names, in column order.
    :param within_k: Tolerance for the within-k accuracy metric.

    :returns: One row per ``(variable, path)`` pair, columns as in :data:`_METRIC_COLUMNS`.
    """
    rows = []
    for path_name, predictions in (("regr", regr), ("cross", cross)):
        metrics = per_variable_regression_metrics(
            orig, predictions, keys, ABCD_INT_FEATURE_INDICES, within_k
        )
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
    return rows


def _print_metrics_rows(rows: list[list]) -> None:
    """Print metrics rows as a plain-text table on stdout."""
    header = " | ".join(_METRIC_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            " | ".join(
                "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v) for v in row
            )
        )


def _render_metrics_table_figure(tables: list[tuple[str, list[list]]], run_id: str) -> plt.Figure:
    """
    Render one metrics table per path side by side in a single matplotlib figure.

    :param tables: ``(title, rows)`` pairs, one per path, rendered left to right in order.
    :param run_id: wandb run id, printed as the report's title-page header.

    :returns: The created figure. Caller owns it and is responsible for closing it.
    """
    max_rows = max(len(rows) for _, rows in tables)
    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 0.4 * max_rows + 3))
    for ax, (title, rows) in zip(axes, tables, strict=True):
        ax.axis("off")
        cell_text = [
            ["" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
            for row in rows
        ]
        table = ax.table(
            cellText=cell_text, colLabels=_METRIC_COLUMNS, loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(_METRIC_COLUMNS))))
        ax.set_title(title, fontweight="bold")
    fig.suptitle(f"Run ID: {run_id}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _build_pdf_report(
    pdf_path: Path,
    rows: list[list],
    orig: np.ndarray,
    regr: np.ndarray,
    cross: np.ndarray,
    keys: list[str],
    run_id: str,
) -> None:
    """
    Render the metrics tables and every per-variable scatter/residual plot into one PDF.

    Each page is added straight from its matplotlib ``Figure`` via ``PdfPages``, so pages stay
    vector graphics instead of being rasterised the way a per-image upload would be.

    :param pdf_path: Destination path for the PDF file.
    :param rows: Metric rows, as returned by :func:`_compute_metrics_rows`.
    :param orig: ``(N, 9)`` ground-truth config values.
    :param regr: ``(N, 9)`` config-encoder self-reconstruction predictions.
    :param cross: ``(N, 9)`` graph-encoder -> theta cross-modal predictions.
    :param keys: Variable names, in column order.
    :param run_id: wandb run id, printed on the report's title page.
    """
    with PdfPages(pdf_path) as pdf:
        tables = [
            (
                f"Per-variable regression metrics -- {path_name}",
                [r for r in rows if r[1] == path_name],
            )
            for path_name in ("regr", "cross")
        ]
        table_fig = _render_metrics_table_figure(tables, run_id=run_id)
        pdf.savefig(table_fig)
        plt.close(table_fig)

        for path_name, predictions in (("regr", regr), ("cross", cross)):
            for i, key in enumerate(keys):
                label = f"{path_name}/{key}"

                scatter_fig = scatter_pred_vs_true(orig[:, i], predictions[:, i], label)
                pdf.savefig(scatter_fig)
                plt.close(scatter_fig)

                residual_fig = residual_histogram(orig[:, i], predictions[:, i], label)
                pdf.savefig(residual_fig)
                plt.close(residual_fig)


def main() -> None:
    """Fetch predictions for a run, compute diagnostics, and re-upload them to the same run."""
    args = _parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    columns, data = _fetch_predictions_table(run)
    orig, regr, cross, keys, meta = _reshape_predictions(columns, data)

    rows = _compute_metrics_rows(orig, regr, cross, keys, args.within_k)
    _print_metrics_rows(rows)

    report_dir_ctx = (
        nullcontext(args.dump_dir) if args.dump_dir is not None else tempfile.TemporaryDirectory()
    )
    with report_dir_ctx as report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = report_dir / "analysis_report.pdf"
        _build_pdf_report(pdf_path, rows, orig, regr, cross, keys, run_id=run.id)

        with (report_dir / "regression_metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_METRIC_COLUMNS)
            writer.writerows(rows)

        with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="must") as wrt_t:
            wrt_t.log({"test/regression_metrics": wandb.Table(_METRIC_COLUMNS, rows)})
            wrt_t.save(str(pdf_path), base_path=str(report_dir), policy="now")
            wrt_t.log_artifact(
                artifact_or_path=str(pdf_path), name=f"test-report-{run.id}", type="report"
            )


if __name__ == "__main__":
    main()
