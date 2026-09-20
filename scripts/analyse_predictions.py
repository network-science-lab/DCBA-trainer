"""Per-variable regression diagnostics for a supcon or baseline test run, from wandb or local."""

import argparse
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
    "mode",
    "r2",
    "pearson_r",
    "relative_error",
    "nrmse",
    "within_k_accuracy",
]


#: Fallback run path used when neither ``run_path`` nor ``--predictions-file`` is given.
_DEFAULT_RUN_PATH = "network-science-lab/DCBA/9dbudg6f"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path",
        nargs="?",
        default=None,
        help="wandb run path, e.g. entity/project/run_id. Mutually exclusive with "
        f"--predictions-file; defaults to {_DEFAULT_RUN_PATH} if neither is given.",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help="Local *.table.json file (as written by scripts/compute_test_predictions.py) to "
        "analyse instead of fetching from wandb. Mutually exclusive with run_path; the "
        "predictions table itself is never fetched from wandb, but the resulting report is "
        "still uploaded to the source run named in the file's meta block, if any.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Analyse the 'predictions-<run_id>-<dataset>' W&B artifact for this dataset "
        "(as logged by scripts/compute_test_predictions.py) instead of the run's own test "
        "split. Only valid together with run_path; mutually exclusive with --predictions-file.",
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
    args = parser.parse_args()
    if args.run_path is not None and args.predictions_file is not None:
        parser.error("run_path and --predictions-file are mutually exclusive")
    if args.dataset is not None and args.predictions_file is not None:
        parser.error("--dataset and --predictions-file are mutually exclusive")
    if args.run_path is None and args.predictions_file is None:
        args.run_path = _DEFAULT_RUN_PATH
    return args


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


def _fetch_predictions_artifact(run: Run, dataset: str) -> tuple[list[str], list[list], dict]:
    """
    Download the ``predictions-<run_id>-<dataset>`` artifact logged by compute_test_predictions.py.

    :param run: The W&B run the artifact was logged on.
    :param dataset: Dataset name the artifact was computed against, e.g. ``"abcd-borderline"``.

    :returns: Tuple of ``(columns, data, meta)`` as stored in the artifact's JSON file.
    """
    artifact_path = f"{run.entity}/{run.project}/predictions-{run.id}-{dataset}:latest"
    artifact = wandb.Api().artifact(artifact_path)
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_dir = Path(artifact.download(root=tmp_dir))
        table_path = next(artifact_dir.rglob("*.table.json"))
        with table_path.open(encoding="utf-8") as f:
            table_json = json.load(f)

    return table_json["columns"], table_json["data"], table_json.get("meta", {})


def _load_local_predictions_table(path: Path) -> tuple[list[str], list[list], dict]:
    """
    Load a local ``*.table.json`` predictions file, matching :func:`_fetch_predictions_table`.

    Reads the file written by ``scripts/compute_test_predictions.py``, which uses the same
    ``{"columns": [...], "data": [...]}`` schema as the wandb run-table artifact (see
    :func:`~dcba.eval.predictions_table.build_predictions_table`), so no downstream reshaping
    logic needs to change, plus a ``meta`` block naming the run and dataset it was computed from.

    :param path: Path to the local JSON file.

    :returns: Tuple of ``(columns, data, meta)`` as stored in the file; ``meta`` is empty for a
        file written before that block existed.
    """
    with path.open(encoding="utf-8") as f:
        table_json = json.load(f)
    return table_json["columns"], table_json["data"], table_json.get("meta", {})


def _source_labels(meta: dict, fallback: str) -> tuple[str, str]:
    """
    Build the human-readable title and the file-name slug identifying the analysed predictions.

    Both name the run *and* the dataset, so a report cannot be mistaken for one computed on a
    different dataset once it is moved out of its directory or uploaded.

    :param meta: Metadata block describing the source, as written by
        ``scripts/compute_test_predictions.py``'s ``_build_meta`` (may be empty or partial).
    :param fallback: Identifier to fall back on when ``meta`` names no run -- the wandb run id, or
        the predictions file's stem.

    :returns: Tuple of ``(title, slug)``; the title reads e.g.
        ``9dbudg6f -- abcd-borderline (whole dataset)`` and the slug ``9dbudg6f-abcd-borderline``.
    """
    run_id = meta.get("run_id") or fallback
    dataset = meta.get("dataset")
    if dataset is None:
        return run_id, run_id

    scope = {"own_test_split": "own test split", "whole_dataset": "whole dataset"}.get(
        meta.get("test_scope", "")
    )
    title = f"{run_id} -- {dataset}" + (f" ({scope})" if scope else "")
    return title, f"{run_id}-{dataset}"


def _reshape_predictions(
    columns: list[str], data: list[list]
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], pd.DataFrame]:
    """
    Reshape the flat ``{i}-orig`` / ``{i}-<mode>`` table rows into arrays.

    Groups rows by whatever kind suffix follows each sample id, so this adapts to however many
    prediction paths a wrapper logs alongside ``orig`` -- the supcon wrapper logs ``regr`` and
    ``crsm``, the baseline wrapper logs a single ``pred``.

    :param columns: Table column names, as produced by a wrapper's ``on_test_epoch_end`` (e.g.
        :meth:`~dcba.wrappers.supcon.DCBASupConWrapper.on_test_epoch_end`) --
        ``["sample", "instance", "replica", <one column per ABCD variable>]``.
    :param data: Table rows in the same column order.

    :returns: Tuple ``(orig, predictions, keys, meta)`` where ``orig`` is a ``(N, 9)``
        ground-truth array, ``predictions`` maps each non-``orig`` kind found in the table to
        its own ``(N, 9)`` array aligned by sample, ``keys`` are the variable names with any
        ``_norm`` suffix stripped, and ``meta`` is a DataFrame with ``instance``/``replica``
        columns, indexed by the same row position as the arrays.
    """
    variable_columns = [c for c in columns if c not in ("sample", "instance", "replica")]
    keys = [c[: -len("_norm")] if c.endswith("_norm") else c for c in variable_columns]
    var_start = len(columns) - len(variable_columns)
    instance_idx = columns.index("instance")
    replica_idx = columns.index("replica")

    rows_by_kind: dict[str, dict[str, list[float]]] = {}
    metadata: dict[str, tuple[str, int]] = {}
    for row in data:
        sample_id, kind = row[0].rsplit("-", 1)
        rows_by_kind.setdefault(kind, {})[sample_id] = row[var_start:]
        if kind == "orig":
            metadata[sample_id] = (row[instance_idx], row[replica_idx])

    sample_ids = sorted(rows_by_kind["orig"], key=int)
    orig = np.array([rows_by_kind["orig"][i] for i in sample_ids], dtype=float)
    predictions = {
        kind: np.array([rows_by_kind[kind][i] for i in sample_ids], dtype=float)
        for kind in rows_by_kind
        if kind != "orig"
    }
    meta = pd.DataFrame([metadata[i] for i in sample_ids], columns=["instance", "replica"])
    return orig, predictions, keys, meta


def _compute_metrics_df(
    orig: np.ndarray, predictions: dict[str, np.ndarray], keys: list[str], within_k: int
) -> pd.DataFrame:
    """
    Compute per-variable regression metrics for every prediction path.

    :param orig: ``(N, 9)`` ground-truth config values.
    :param predictions: Maps each prediction path's mode name (e.g. ``regr``, ``crsm``, ``pred``)
        to its ``(N, 9)`` array of predicted values.
    :param keys: Variable names, in column order.
    :param within_k: Tolerance for the within-k accuracy metric.

    :returns: DataFrame with one row per ``(variable, mode)`` pair, columns as in
        :data:`_METRIC_COLUMNS`.
    """
    rows = []
    for mode, preds in predictions.items():
        metrics = per_variable_regression_metrics(
            orig, preds, keys, ABCD_INT_FEATURE_INDICES, within_k
        )
        for key in keys:
            m = metrics[key]
            rows.append(
                {
                    "variable": key,
                    "mode": mode,
                    "r2": m["r2"],
                    "pearson_r": m["pearson_r"],
                    "relative_error": m["relative_error"],
                    "nrmse": m["nrmse"],
                    "within_k_accuracy": m.get("within_k_accuracy"),
                }
            )
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def _print_metrics_df(metrics_df: pd.DataFrame) -> None:
    """Print the metrics DataFrame as a plain-text table on stdout."""
    print(metrics_df.to_string(index=False, na_rep=""))


def _render_metrics_table_figure(
    tables: list[tuple[str, pd.DataFrame]], source_label: str
) -> plt.Figure:
    """
    Render one metrics table per path side by side in a single matplotlib figure.

    :param tables: ``(title, metrics_df)`` pairs, one per path, rendered left to right in order.
    :param source_label: Run and dataset the metrics were computed from, printed as the report's
        title-page header.

    :returns: The created figure. Caller owns it and is responsible for closing it.
    """
    max_rows = max(len(df) for _, df in tables)
    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 0.4 * max_rows + 3))
    axes = np.atleast_1d(axes)
    for ax, (title, df) in zip(axes, tables, strict=True):
        ax.axis("off")
        cell_text = [
            ["" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else str(v) for v in row]
            for row in df.itertuples(index=False)
        ]
        table = ax.table(
            cellText=cell_text, colLabels=list(df.columns), loc="center", cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        ax.set_title(title, fontweight="bold")
    fig.suptitle(f"Source: {source_label}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _resolve_upload_run(run_path: str | None) -> Run | None:
    """
    Fetch the W&B run a local predictions file names, for uploading the report to.

    Only called when analysing a local ``--predictions-file`` -- makes no wandb call at all when
    the file's ``meta`` block has no ``run_path`` (e.g. a file written before that field existed).

    :param run_path: Full ``entity/project/run_id`` path from the file's ``meta`` block, or
        ``None`` if the file names no run.

    :returns: The resolved run, or ``None`` if ``run_path`` is ``None``.
    """
    return wandb.Api().run(run_path) if run_path is not None else None


def _upload_report(
    run: Run,
    metrics_df: pd.DataFrame,
    pdf_path: Path,
    report_dir: Path,
    dataset: str | None,
    slug: str,
) -> None:
    """
    Log the metrics table and PDF report to ``run``'s W&B history and artifacts.

    The metrics table is logged under a dataset-specific key rather than the fixed
    ``test/regression_metrics``, so evaluating one run against several datasets (e.g. its own
    ``abcd-big`` test split and the held-out ``abcd-borderline`` set) doesn't have the second
    evaluation silently overwrite the first in the run's history.

    :param run: The W&B run to attach the report to.
    :param metrics_df: Metrics, as returned by :func:`_compute_metrics_df`.
    :param pdf_path: Local path of the rendered PDF report.
    :param report_dir: Directory ``pdf_path`` lives under, used as the W&B save base path.
    :param dataset: Name of the dataset the predictions were computed against, or ``None`` if it
        could not be determined -- falls back to the fixed, undated key in that case.
    :param slug: Run+dataset identifier used to name the uploaded report artifact.
    """
    metric_key = f"test/regression_metrics_{dataset}" if dataset else "test/regression_metrics"
    with wandb.init(id=run.id, project=run.project, entity=run.entity, resume="must") as wrt_t:
        wrt_t.log({metric_key: wandb.Table(dataframe=metrics_df)})
        wrt_t.save(str(pdf_path), base_path=str(report_dir), policy="now")
        wrt_t.log_artifact(
            artifact_or_path=str(pdf_path), name=f"test-report-{slug}", type="report"
        )


def build_pdf_report(
    pdf_path: Path,
    metrics_df: pd.DataFrame,
    orig: np.ndarray,
    predictions: dict[str, np.ndarray],
    keys: list[str],
    source_label: str,
) -> None:
    """
    Render the metrics tables and every per-variable scatter/residual plot into one PDF.

    Each page is added straight from its matplotlib ``Figure`` via ``PdfPages``, so pages stay
    vector graphics instead of being rasterised the way a per-image upload would be.

    :param pdf_path: Destination path for the PDF file.
    :param metrics_df: Metrics, as returned by :func:`_compute_metrics_df`.
    :param orig: ``(N, 9)`` ground-truth config values.
    :param predictions: Maps each prediction path's mode name to its ``(N, 9)`` array of
        predicted values, as returned by :func:`_reshape_predictions`.
    :param keys: Variable names, in column order.
    :param source_label: Run and dataset the metrics were computed from, printed on the report's
        title page.
    """
    with PdfPages(pdf_path) as pdf:
        tables = [
            (
                f"Per-variable regression metrics -- {mode}",
                metrics_df[metrics_df["mode"] == mode],
            )
            for mode in predictions
        ]
        table_fig = _render_metrics_table_figure(tables, source_label=source_label)
        pdf.savefig(table_fig)
        plt.close(table_fig)

        for mode, preds in predictions.items():
            for i, key in enumerate(keys):
                label = f"{mode}/{key}"

                scatter_fig = scatter_pred_vs_true(orig[:, i], preds[:, i], label)
                pdf.savefig(scatter_fig)
                plt.close(scatter_fig)

                residual_fig = residual_histogram(orig[:, i], preds[:, i], label)
                pdf.savefig(residual_fig)
                plt.close(residual_fig)


def main() -> None:
    """
    Compute per-variable regression diagnostics for a run's or a local file's test predictions.

    Given ``--predictions-file``, reads a local JSON file and never fetches from wandb, but still
    uploads the resulting report to the source run named in the file's ``meta`` block (skipped if
    that run is unknown). Given ``--dataset``, fetches that dataset's ``predictions-<run_id>-
    <dataset>`` artifact for ``run_path`` instead of the run's own test split. Otherwise fetches
    the run's own ``test/predictions`` table for ``run_path``. Either wandb path re-uploads the
    report to that same run.
    """
    args = _parse_args()

    if args.predictions_file is not None:
        run = None
        columns, data, source_meta = _load_local_predictions_table(args.predictions_file)
        fallback = args.predictions_file.stem
    else:
        api = wandb.Api()
        run = api.run(args.run_path)
        if args.dataset is not None:
            columns, data, source_meta = _fetch_predictions_artifact(run, args.dataset)
            source_meta.setdefault("run_path", args.run_path)
            fallback = f"{run.id}-{args.dataset}"
        else:
            columns, data = _fetch_predictions_table(run)
            # The wandb table is logged by the run's own test loop, so it is always that run's
            # held-out split of the dataset it was trained on.
            source_meta = {
                "run_id": run.id,
                "run_path": args.run_path,
                "dataset": Path(run.config["data"]["dataset_root"]).name,
                "test_scope": "own_test_split",
            }
            fallback = run.id
    source_label, slug = _source_labels(source_meta, fallback)
    dataset = source_meta.get("dataset")

    orig, predictions, keys, meta = _reshape_predictions(columns, data)

    print(f"Source: {source_label}")
    metrics_df = _compute_metrics_df(orig, predictions, keys, args.within_k)
    _print_metrics_df(metrics_df)

    report_dir_ctx = (
        nullcontext(args.dump_dir) if args.dump_dir is not None else tempfile.TemporaryDirectory()
    )
    with report_dir_ctx as report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        # Every file carries the run and dataset in its name, so a report stays identifiable once
        # it is copied out of its directory or uploaded next to another dataset's report.
        pdf_path = report_dir / f"analysis_report_{slug}.pdf"
        build_pdf_report(pdf_path, metrics_df, orig, predictions, keys, source_label=source_label)

        metrics_df.to_csv(
            report_dir / f"regression_metrics_{slug}.csv", index=False, encoding="utf-8"
        )
        metrics_df.to_latex(
            report_dir / f"regression_metrics_{slug}.tex", index=False, encoding="utf-8"
        )

        upload_run = run if run is not None else _resolve_upload_run(source_meta.get("run_path"))
        if upload_run is None:
            print(f"No source run recorded; report written to {report_dir} only")
        else:
            _upload_report(upload_run, metrics_df, pdf_path, report_dir, dataset, slug)
            print(f"Report uploaded to {upload_run.id}; also written to {report_dir}")


if __name__ == "__main__":
    main()
