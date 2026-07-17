# ruff: noqa
"""Derive per-feature ABCDConstraintPenaltyLoss weights from a run's regr/cross regression metrics.

`configs/gps-ae-supcon-output-weights.yaml`'s `training.losses.reg.args.weights` implements
`weight_i = 1 + clip(regr_R2 - cross_R2, 0, 1)`: give more
gradient priority to parameters where the graph encoder ("cross": theta decoded from `z_g`) loses
the most R2 relative to the config autoencoder alone ("regr": theta decoded from `z_theta`), so the
loss pushes hardest on parameters the *encoder* -- not just the decoder -- currently recovers worst.
That vector was derived by hand from a wandb table read off one run; this script automates it for
any run, in `ABCD_CONFIG_KEYS` order, ready to paste into a config's `weights` field.

Usage:
    # no args: fetches the default run_path below fresh from wandb
    uv run scripts/derive_output_weights.py

    # fetches test/predictions fresh from wandb for a specific run
    uv run scripts/derive_output_weights.py network-science-lab/DCBA/3or3wpqk

    # reuses a regression_metrics.csv already written by `analyse_predictions.py --dump-dir`
    uv run scripts/derive_output_weights.py --csv .analysis/report/regression_metrics.csv
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from wandb.apis.public import Run

from dcba.dataset import ABCD_CONFIG_KEYS, ABCD_INT_FEATURE_INDICES
from dcba.training.metrics import per_variable_regression_metrics

GAP_CLIP = (0.0, 1.0)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path",
        nargs="?",
        default="network-science-lab/DCBA/2kolqt2w",
        help="wandb run path, e.g. entity/project/run_id. Ignored if --csv is given.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to a regression_metrics.csv already written by "
        "'analyse_predictions.py --dump-dir'. Skips fetching from wandb entirely.",
    )
    parser.add_argument(
        "--within-k",
        type=int,
        default=10,
        help="Tolerance for the within-k accuracy metric (only computed when fetching fresh from "
        "wandb; unused by the weight formula itself, which only reads r2).",
    )
    return parser.parse_args()


def _fetch_predictions_table(run: Run) -> tuple[list[str], list[list]]:
    """
    Download the ``test/predictions`` wandb.Table logged during testing for a run.

    Mirrors ``analyse_predictions.py``'s helper of the same name -- duplicated rather than
    imported since ``scripts/`` has no ``__init__.py`` (each script is a standalone CLI, not an
    importable package member).

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Reshape the flat ``{i}-orig`` / ``{i}-regr`` / ``{i}-crsm`` table rows into arrays.

    :param columns: Table column names.
    :param data: Table rows in the same column order.

    :returns: ``(orig, regr, cross, keys)`` -- the first three ``(N, 9)`` arrays aligned by
        sample, ``keys`` the variable names with any ``_norm`` suffix stripped.
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


def _metrics_from_wandb(run_path: str, within_k: int) -> pd.DataFrame:
    """
    Fetch a run's ``test/predictions`` table and recompute per-variable regression metrics.

    :param run_path: wandb run path, e.g. ``entity/project/run_id``.
    :param within_k: Tolerance for the within-k accuracy metric.

    :returns: Long-format DataFrame with one row per ``(variable, mode)`` pair and an ``r2``
        column -- a subset of ``analyse_predictions.py``'s ``regression_metrics.csv`` shape.
    """
    api = wandb.Api()
    run = api.run(run_path)
    columns, data = _fetch_predictions_table(run)
    orig, regr, cross, keys = _reshape_predictions(columns, data)

    rows = []
    for mode, predictions in (("regr", regr), ("cross", cross)):
        metrics = per_variable_regression_metrics(
            orig, predictions, keys, ABCD_INT_FEATURE_INDICES, within_k
        )
        for key in keys:
            rows.append({"variable": key, "mode": mode, "r2": metrics[key]["r2"]})
    return pd.DataFrame(rows)


def _derive_weights(r2: pd.Series) -> list[float]:
    """
    Compute ``weight_i = 1 + clip(regr_R2 - cross_R2, 0, 1)`` per :data:`ABCD_CONFIG_KEYS`.

    :param r2: Series indexed by ``(variable, mode)`` with ``mode`` in ``{"regr", "cross"}``.

    :returns: One weight per feature, in :data:`ABCD_CONFIG_KEYS` order.
    """
    return [
        1.0 + float(np.clip(r2[(key, "regr")] - r2[(key, "cross")], *GAP_CLIP))
        for key in ABCD_CONFIG_KEYS
    ]


def main() -> None:
    """Derive and print ABCDConstraintPenaltyLoss weights, ready to paste into a config."""
    args = _parse_args()

    if args.csv is not None:
        metrics_df = pd.read_csv(args.csv)
    else:
        metrics_df = _metrics_from_wandb(args.run_path, args.within_k)

    r2 = metrics_df.set_index(["variable", "mode"])["r2"]
    weights = _derive_weights(r2)

    print(f"{'variable':<8} {'regr_r2':>10} {'cross_r2':>10} {'gap':>8} {'weight':>8}")
    for key, weight in zip(ABCD_CONFIG_KEYS, weights):
        regr_r2, cross_r2 = r2[(key, "regr")], r2[(key, "cross")]
        gap = regr_r2 - cross_r2
        print(f"{key:<8} {regr_r2:>10.4f} {cross_r2:>10.4f} {gap:>8.4f} {weight:>8.4f}")

    print(f"\nweights: [{', '.join(f'{w:.3f}' for w in weights)}]")


if __name__ == "__main__":
    main()
