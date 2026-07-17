# ruff: noqa
"""Recommend a `num_clusters` budget for `GPSEncoder`'s soft-clustering pool from real community counts.

`GPSEncoder(..., num_clusters=K)` (`src/dcba/models/gps_encoder.py`) assigns nodes to a fixed
budget of `K` learned slots. `K` is meant to be a generous upper bound on
how many communities a graph in the dataset actually has -- too tight a budget forces multiple real
communities to share one slot, corrupting exactly the size-distribution-shape statistics
(`c_max`, `t2`) the branch exists to recover. The current default (`num_clusters=16`, set in
`configs/gps-ae-supcon-output-weights.yaml`) was chosen as "a generous ceiling" without measuring
the dataset's actual community-count distribution.

This script loads a random sample of graphs (ground-truth `data["actor"].community` is generator
metadata, safe to use for a one-off dataset-statistics script even though the model itself must
never see it), counts the real communities per replica, and reports the distribution so a
data-backed `K` can be picked instead of a guess.

Usage:
    uv run scripts/recommend_num_clusters.py
    uv run scripts/recommend_num_clusters.py --max-samples 2000 --percentile 99

To check a different dataset, pass `--dataset`.
"""

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dcba.dataset import ABCDDataset, ConstantNodeFeatures
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
    }
)
PLOT_DPI = 300
OUTPUT_PATH = Path(".analysis/num_clusters_recommendation.png")
SAFETY_MARGIN = 1.2


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="abcd-big", help="Dataset directory name under DATA_ROOT."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Number of replicas to sample (graphs are loaded lazily; sampling avoids paying the "
        "full-dataset I/O cost of a multi-GB ABCD dataset just to read community labels).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile of the observed per-graph community-count distribution to base the "
        "num_clusters recommendation on.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replica sampling.")
    return parser.parse_args()


def _community_counts(dataset: ABCDDataset, max_samples: int) -> np.ndarray:
    """
    Count real (non-zero) communities per sampled graph replica.

    `ABCDDataset` shuffles its replica list once at construction and loads graphs lazily on
    `__getitem__`, so taking the first `max_samples` items is a random sample without paying the
    I/O cost of loading the whole dataset.

    For multi-layer (mABCD) graphs, the max across layers is taken -- the clustering branch pools
    all layers' node embeddings into one shared assignment (see
    `GPSEncoder._cluster_pool`), so the binding constraint is the layer with the most communities.

    :param dataset: An :class:`~dcba.dataset.ABCDDataset` instance.
    :param max_samples: Number of replicas to sample from the front of the (already shuffled) list.

    :returns: ``(min(max_samples, len(dataset)),)`` array of per-graph community counts.
    """
    n = min(max_samples, len(dataset))
    counts = np.empty(n, dtype=np.int64)
    for i in range(n):
        community = dataset[i]["actor"].community  # (N, L), long
        per_layer = [int(community[:, l].unique().numel()) for l in range(community.shape[1])]
        # A `0` label marks an inactive (non-mABCD) node, not a real community -- exclude it,
        # but only subtract when actually present so single-layer ABCD graphs (no zero label)
        # aren't undercounted by one.
        has_inactive = [bool((community[:, l] == 0).any()) for l in range(community.shape[1])]
        counts[i] = max(c - 1 if inactive else c for c, inactive in zip(per_layer, has_inactive))
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{n} graphs.")
    return counts


def _plot_distribution(counts: np.ndarray, recommendation: int, out_path: Path) -> None:
    """
    Plot a histogram of per-graph community counts with the recommended `num_clusters` marked.

    :param counts: Per-graph community counts, as returned by :func:`_community_counts`.
    :param recommendation: The recommended `num_clusters` value.
    :param out_path: PNG output path; parent directory is created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    sns.histplot(counts, bins=30, color="steelblue", ax=ax)
    ax.axvline(
        recommendation,
        color="firebrick",
        linestyle="--",
        linewidth=1.5,
        label=f"recommended num_clusters = {recommendation}",
    )
    ax.set_xlabel("Real communities per graph")
    ax.set_ylabel("Count")
    ax.set_title("ABCD Community-Count Distribution vs. Recommended num_clusters")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Recommend `num_clusters` for `GPSEncoder` from real community counts on a dataset."""
    args = _parse_args()
    random.seed(args.seed)

    dataset = ABCDDataset.from_dataset(
        DATA_ROOT / args.dataset, node_transform=ConstantNodeFeatures()
    )
    logger.info(f"Sampling up to {args.max_samples} of {len(dataset)} replicas.")
    counts = _community_counts(dataset, args.max_samples)

    pct = float(np.percentile(counts, args.percentile))
    recommendation = int(np.ceil(pct * SAFETY_MARGIN))

    logger.info(
        f"Community count -- min: {counts.min()}, median: {np.median(counts):.1f}, "
        f"p{args.percentile:.0f}: {pct:.1f}, max: {counts.max()}"
    )
    logger.info(
        f"Recommended num_clusters: {recommendation} "
        f"(p{args.percentile:.0f}={pct:.1f} x {SAFETY_MARGIN} safety margin)."
    )

    _plot_distribution(counts, recommendation, OUTPUT_PATH)
    logger.info(f"Wrote distribution chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
