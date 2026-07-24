# ruff: noqa
"""Check whether ABCDConfigScaler's bounds match each parameter's real dynamic range.

:class:`~dcba.dataset.scalers.ABCDConfigScaler` maps every ABCD parameter to ``[0, 1]`` using
:data:`~dcba.dataset.scalers.ABCD_PARAM_BOUNDS`. Six of the nine parameters
(``n, c_min, c_max, d_min, d_max, nout``) share the same analytical upper bound (``n_max``), even
though their realised values in an actual dataset can occupy wildly different fractions of that
shared bound. A parameter that ends up compressed into a tiny sliver of ``[0, 1]`` gets almost no
gradient signal from any loss computed on the scaled config (regression loss, contrastive
negative-weighting distance) -- the model effectively cannot learn it, independent of encoder
architecture.

This script loads a dataset's raw (unscaled) configs, computes each parameter's realised standard
deviation as a percentage of its scaler bound width, and flags parameters below
:data:`STARVED_THRESHOLD_PCT` as starved. It requires no trained model or W&B access -- only the
dataset -- so it can be re-run after any change to :data:`ABCD_PARAM_BOUNDS` to check the fix
before spending time on training.

Usage:
    uv run scripts/check_config_scaling.py

To check a different dataset, edit :data:`DATASET_NAME` at the top of this file.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dcba_data_set.graph_io import load_dataset
from dcba_data_set.graph_io.data_models import DCBAInstanceConfig
from torch import Tensor

from dcba.dataset.scalers import ABCD_CONFIG_KEYS, ABCD_PARAM_BOUNDS, ABCDConfigToTensor
from dcba.utils.paths import DATA_ROOT

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)
PLOT_DPI = 300

DATASET_NAME = "abcd-big"
OUTPUT_PATH = Path(".analysis/param_scaling_occupancy.png")
#: Below this % of its scaler bound width actually used, flag a parameter as starved.
STARVED_THRESHOLD_PCT = 2.0

#: Human-readable, LaTeX-formatted label for each raw ABCD parameter (see ABCD_CONFIG_KEYS).
ABCD_PARAM_LABELS = {
    "n": "$n$",
    "t1": r"$\tau_1$",
    "t2": r"$\tau_2$",
    "xi": r"$\xi$",
    "c_min": "$c_{min}$",
    "c_max": "$c_{max}$",
    "d_min": "$d_{min}$",
    "d_max": "$d_{max}$",
    "nout": "$n_{out}$",
}


def _load_raw_configs(dataset_root: Path) -> Tensor:
    """
    Load every instance's raw (unscaled) ABCD config as a ``(N, 9)`` tensor.

    :param dataset_root: Root directory of the dataset (flat or chunked layout), as accepted by
        :func:`~dcba_data_set.graph_io.load_dataset`.

    :returns: ``(N, 9)`` float tensor, one row per instance, columns ordered by
        :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS`.
    """
    records = load_dataset(dataset_root)
    to_tensor = ABCDConfigToTensor()
    rows = [to_tensor(DCBAInstanceConfig.from_instance_record(r)) for r in records]
    logger.info(f"Loaded {len(rows)} instance configs from {dataset_root}.")
    return torch.stack(rows)


def _occupied_range_pct(raw_configs: Tensor) -> dict[str, float]:
    """
    Compute, per parameter, how much of its scaler bound width the dataset actually uses.

    Uses the realised standard deviation rather than the min/max span so a handful of outliers
    cannot make a parameter look better-covered than it typically is.

    :param raw_configs: ``(N, 9)`` raw config tensor, as returned by :func:`_load_raw_configs`.

    :returns: Dict mapping each key in :data:`~dcba.dataset.scalers.ABCD_CONFIG_KEYS` to the
        percentage of its scaler bound width covered by one standard deviation of real data.
    """
    bounds = ABCD_PARAM_BOUNDS
    c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
    c_max_idx = ABCD_CONFIG_KEYS.index("c_max")

    # c_min is scaled as c_min / c_max (see ABCDConfigScaler), not against a fixed bound like
    # every other feature -- its "raw" value for occupancy purposes is that ratio, not the count.
    values = raw_configs.clone()
    values[:, c_min_idx] = raw_configs[:, c_min_idx] / raw_configs[:, c_max_idx].clamp(min=1.0)

    std = values.std(dim=0, unbiased=False)
    occupied = {}
    for i, key in enumerate(ABCD_CONFIG_KEYS):
        lo, hi = bounds[key]
        occupied[key] = 100.0 * std[i].item() / (hi - lo)
    return occupied


def _plot_occupancy(occupied_pct: dict[str, float], out_path: Path) -> None:
    """
    Plot each parameter's scaler-bound occupancy as a sorted horizontal bar chart.

    :param occupied_pct: Per-parameter occupancy percentages, as returned by
        :func:`_occupied_range_pct`.
    :param out_path: PNG output path; parent directory is created if missing.
    """
    keys_sorted = sorted(occupied_pct, key=lambda k: occupied_pct[k])
    values = [occupied_pct[k] for k in keys_sorted]
    colors = ["firebrick" if v < STARVED_THRESHOLD_PCT else "steelblue" for v in values]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([ABCD_PARAM_LABELS[k] for k in keys_sorted], values, color=colors)
    ax.axvline(STARVED_THRESHOLD_PCT, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Occupied Scaled Range (Std / Bound Width, %)")
    ax.set_title("How Much of Its [0, 1] Scaled Range Each ABCD Parameter Actually Uses")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Check config scaling occupancy for :data:`DATASET_NAME` and report starved parameters."""
    raw_configs = _load_raw_configs(DATA_ROOT / DATASET_NAME)
    occupied_pct = _occupied_range_pct(raw_configs)

    logger.info(f"{'param':<8} {'occupied_%':>12}")
    starved = []
    for key in sorted(occupied_pct, key=lambda k: occupied_pct[k]):
        pct = occupied_pct[key]
        logger.info(f"{key:<8} {pct:>11.3f}%")
        if pct < STARVED_THRESHOLD_PCT:
            starved.append(key)

    if starved:
        logger.warning(
            f"Starved parameters (< {STARVED_THRESHOLD_PCT}% of their scaled range): "
            f"{', '.join(starved)}. These get almost no gradient from any loss computed on the "
            "scaled config and are effectively unlearnable regardless of encoder architecture."
        )
    else:
        logger.info(f"No parameters below the {STARVED_THRESHOLD_PCT}% starved threshold.")

    _plot_occupancy(occupied_pct, OUTPUT_PATH)
    logger.info(f"Wrote occupancy chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
