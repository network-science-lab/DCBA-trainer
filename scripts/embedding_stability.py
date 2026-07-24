# ruff: noqa
"""Embedding stability analysis for gps-ae-supcon runs logged to Weights & Biases.

For each run in :data:`RUNS`, rebuilds the trained graph encoder (``z_g``) and config encoder
(``z_theta``) from the run's logged config, downloads its best-epoch model checkpoint artifact,
and runs both encoders over a sample of the run's test split. It then checks whether distance in
ABCD parameter space (``d_theta``) is reflected in the learned embedding spaces:

- ``dtheta_vs_dzg.png`` / ``dtheta_vs_dztheta.png``: pairwise ``d_theta`` vs embedding distance,
  binned mean trend with a percentile band. A flat trend means the embedding does not
  distinguish configs that differ a lot; a rising trend means distance is preserved.
- ``stability_{zg,ztheta}.png``: distribution of embedding distance for same-instance pairs
  (different graph realisations of one theta) versus all other pairs. Different realisations of
  the same instance are expected to embed close together; this shows whether that holds.
- ``proj_{tsne,umap}_{zg,ztheta}_by_<key>.png``: 2D t-SNE and UMAP projections of the embedding,
  coloured by each raw ABCD parameter, to check whether embeddings cluster along that parameter.
  Both are shown since the two methods emphasise different structure (t-SNE local neighbourhoods,
  UMAP more global layout) and can disagree.
- ``embeddings.pt``: the raw per-graph ``z_g``/``z_theta`` embeddings with their ``instance_id`` /
  ``replica`` matching keys and theta tensors, for downstream reuse (e.g. Mahalanobis distance
  learning over ``z_g``). Not consumed by this script's own plots.

Prerequisites:

- ``wandb login`` (or an existing ``.netrc`` entry) with read access to the run's project.
- The ABCD dataset directory recorded in each run's config (``data.dataset_root``) must already
  exist and be readable on the machine this script runs on -- pull it with ``dvc pull`` first if
  needed. This script does not fetch data itself.

Usage:
    uv run scripts/embedding_stability.py

To analyse different runs, edit :data:`RUNS` at the top of this file.
"""

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap
import wandb
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from dcba.datamodule import ABCDDataModule
from dcba.dataset import ABCDConfigScaler
from dcba.dataset.scalers import ABCD_CONFIG_KEYS
from dcba.training.loss import config_pairwise_distance
from dcba.training.trainer import (
    _build_node_transform,
    _build_scaler,
    _build_transform,
    build_supcon_wrapper,
)
from dcba.utils.paths import DATA_ROOT
from dcba.wrappers.supcon import DCBASupConWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "legend.frameon": False,
    }
)
PLOT_DPI = 300

#: LaTeX math form for each embedding, used in axis/title labels.
EMBED_MATH = {"z_g": r"z_g", "z_theta": r"z_\theta"}

WANDB_ENTITY = "network-science-lab"
WANDB_PROJECT = "dcba"
OUTPUT_DIR = Path(".analysis")
CACHE_DIR = Path(".analysis/checkpoints")
#: Max test instances to sample per run; set to a negative value to use the entire test split.
NUM_INSTANCES = -1
SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJ_AXIS_LIM = 4.5

#: (run_id, human-readable label) for the current gps-ae-supcon comparison set.
RUNS: list[tuple[str, str]] = [
    ("3or3wpqk", "gps-ae-supcon, no-community-features, batch64"),
    ("oraxoj6n", "gps-ae-supcon, no-community-features, batch64"),
]

#: Per-run patches for ``models.graph`` fields that existed in :class:`~dcba.models.gps_encoder.GPSEncoder`
#: at training time but were not yet logged to that run's W&B config -- without this,
#: :func:`~dcba.training.trainer.build_supcon_wrapper` falls back to *today's* class default for
#: the missing field, which can silently mismatch an older checkpoint's shape.
#:
#: ``3or3wpqk`` logged ``num_clusters=16`` (so its checkpoint has the cluster-pooling branch) but
#: no ``cluster_size_quantiles`` -- today's default is a 5-element tuple, giving
#: ``proj_in_dim = 2 * hidden_dim + 5 = 133``, but the checkpoint's ``_proj.0.weight`` is shaped
#: for ``2 * hidden_dim + 2 = 130``. Confirmed by loading the checkpoint with
#: ``strict=False`` after patching to a 2-element tuple: zero missing/unexpected keys.
CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "3or3wpqk": {"cluster_size_quantiles": (0.0, 1.0)},
}


def _apply_config_override(cfg: dict, run_id: str) -> dict:
    """
    Patch ``cfg["models"]["graph"]`` with any :data:`CONFIG_OVERRIDES` entry for ``run_id``.

    :param cfg: Run config as logged to W&B.
    :param run_id: W&B run ID, used to look up :data:`CONFIG_OVERRIDES`.

    :returns: ``cfg`` unchanged if no override is registered for ``run_id``; otherwise a deep
        copy with the override applied.
    """
    overrides = CONFIG_OVERRIDES.get(run_id)
    if not overrides:
        return cfg
    patched = copy.deepcopy(cfg)
    patched["models"]["graph"].update(overrides)
    return patched


@dataclass
class RunEmbeddings:
    """
    Per-sample theta and encoder embeddings collected from one run's test split.

    ``theta_raw`` (human-readable ABCD units) is for per-parameter colouring only
    (:func:`_plot_projection`). ``theta_scaled`` (each parameter linearly mapped to ``[0, 1]``,
    matching what the training loss itself sees) is what :func:`_pairwise_distances` uses, so
    that no single parameter's raw range (e.g. ``n`` in the thousands) dominates the distance.
    ``instance_id`` is the same grouping key :meth:`DCBASupConWrapper._instance_labels` hashes to
    build positive pairs during training -- samples sharing one ``instance_id`` are different
    graph realisations of the same theta ("replicas"). ``(instance_id, replica)`` together
    uniquely identify each graph realisation, so downstream consumers can match a saved ``z_g``
    row back to its exact source graph (see :func:`_save_embeddings`).
    """

    run_id: str
    run_name: str
    theta_raw: Tensor
    theta_scaled: Tensor
    z_g: Tensor
    z_theta: Tensor
    instance_id: list[str]
    replica: Tensor


def _download_checkpoint(run: wandb.apis.public.Run, cache_dir: Path) -> Path:
    """
    Download the run's best model checkpoint artifact (``model-epoch-*``, not ``last.ckpt``).

    :param run: W&B run to fetch the artifact from.
    :param cache_dir: Directory under which the artifact is downloaded (one subdirectory per run).

    :returns: Local path to the downloaded ``.ckpt`` file.
    """
    candidates = [
        a for a in run.logged_artifacts() if a.type == "model" and a.name.startswith("model-epoch")
    ]
    if not candidates:
        raise ValueError(
            f"No best-epoch ('model-epoch-*') model artifact found for run {run.id}. "
            "This usually means the run never reached a validation checkpoint (e.g. a "
            "smoke test with too few epochs) and should not be used for this analysis."
        )
    artifact_dir = Path(candidates[0].download(root=str(cache_dir / run.id)))
    ckpt_files = list(artifact_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No .ckpt file found in downloaded artifact for run {run.id}.")
    return ckpt_files[0]


def _load_wrapper(
    cfg: dict, ckpt_path: Path, device: torch.device
) -> tuple[DCBASupConWrapper, ABCDConfigScaler | None]:
    """
    Rebuild the supcon wrapper's architecture from a run config, then load checkpoint weights.

    :param cfg: Run config as logged to W&B (mirrors the training hydra config).
    :param ckpt_path: Local path to the run's ``.ckpt`` file.
    :param device: Device to load the wrapper onto.

    :returns: The loaded wrapper (in eval mode) and the scaler used to train it.
    """
    training_cfg = cfg["training"]
    if training_cfg["wrapper"] != "supcon":
        raise ValueError(
            f"Unsupported wrapper '{training_cfg['wrapper']}' for run; only 'supcon' "
            "(gps-ae-supcon) is handled by this script."
        )

    scaler = _build_scaler(cfg["data"].get("scaler"))
    wrapper = build_supcon_wrapper(cfg)
    state_dict = torch.load(str(ckpt_path), map_location=device)["state_dict"]
    wrapper.load_state_dict(state_dict)
    wrapper.set_scaler(scaler)
    wrapper.eval()
    wrapper.to(device)
    return wrapper, scaler


@torch.no_grad()
def _collect_embeddings(
    wrapper: DCBASupConWrapper,
    scaler: ABCDConfigScaler | None,
    cfg: dict,
    run_id: str,
    run_name: str,
    num_instances: int,
    device: torch.device,
) -> RunEmbeddings:
    """
    Run the trained encoders over a fixed-seed sample of the run's test split.

    :param wrapper: Loaded :class:`DCBASupConWrapper` in eval mode.
    :param scaler: Scaler used to train the wrapper, for recovering human-readable theta values.
    :param cfg: Run config as logged to W&B.
    :param run_id: W&B run ID, stored on the returned :class:`RunEmbeddings` for labelling.
    :param run_name: Human-readable run name, stored on the returned :class:`RunEmbeddings`.
    :param num_instances: Maximum number of test instances to sample, or a negative value to
        use the entire test split.
    :param device: Device to run inference on.

    :returns: Collected theta values and unit-norm embeddings for the sampled instances.
    """
    data_cfg = cfg["data"]
    # Re-resolve the dataset directory name against this machine's DCBA_DATA_ROOT instead of
    # trusting the absolute path logged from the training machine.
    dataset_root = DATA_ROOT / Path(data_cfg["dataset_root"]).name
    # ABCDDataset.__init__ shuffles its items with the unseeded global `random` module, so
    # reseed it here to make the "fixed-seed sample" below actually reproducible.
    random.seed(SEED)
    datamodule = ABCDDataModule(
        dataset_root=dataset_root,
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        single_replica_per_instance=False,
        seed=cfg.get("random_seed", SEED),
        scaler=scaler,
        transform=_build_transform(data_cfg.get("transform")),
        node_transform=_build_node_transform(data_cfg.get("node_transform")),
    )
    datamodule.setup()
    dataset = datamodule.test_dataloader().dataset

    n = len(dataset)
    k = n if num_instances < 0 else min(num_instances, n)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n, generator=generator)[:k].tolist()
    loader = PyGDataLoader(
        Subset(dataset, indices),
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    theta_chunks: list[Tensor] = []
    z_g_chunks: list[Tensor] = []
    z_theta_chunks: list[Tensor] = []
    instance_ids: list[str] = []
    replica_chunks: list[Tensor] = []
    for batch in tqdm(loader, desc=f"{run_id} ({run_name})", unit="batch"):
        batch = batch.to(device)
        theta = wrapper._unpack_batch(batch)
        z_g, z_theta = wrapper.encode(batch)
        theta_chunks.append(theta.cpu())
        z_g_chunks.append(z_g.cpu())
        z_theta_chunks.append(z_theta.cpu())
        instance_ids.extend(batch.instance_id)
        replica_chunks.append(batch.replica.cpu())

    theta_scaled = torch.cat(theta_chunks)
    theta_raw = scaler.inverse_transform(theta_scaled) if scaler is not None else theta_scaled
    logger.info(f"Collected {theta_raw.size(0)}/{n} test instances for run {run_id}.")
    return RunEmbeddings(
        run_id=run_id,
        run_name=run_name,
        theta_raw=theta_raw,
        theta_scaled=theta_scaled,
        z_g=torch.cat(z_g_chunks),
        z_theta=torch.cat(z_theta_chunks),
        instance_id=instance_ids,
        replica=torch.cat(replica_chunks),
    )


def _pairwise_distances(x: Tensor) -> Tensor:
    """Return flattened upper-triangular pairwise Euclidean distances for ``x``."""
    return torch.pdist(x)


def _pairwise_config_distance(theta_scaled: Tensor) -> Tensor:
    """
    Return flattened upper-triangular pairwise theta distances, ordered like :func:`_pairwise_distances`.

    Uses :func:`~dcba.training.loss.config_pairwise_distance` -- the exact function
    :class:`~dcba.training.loss.MultiPositiveSupConLoss` calls to weight soft negatives -- rather
    than reimplementing the distance independently, so this script cannot silently drift from
    what the training loss actually sees.
    """
    n = theta_scaled.size(0)
    i, j = torch.triu_indices(n, n, offset=1)
    return config_pairwise_distance(theta_scaled, theta_scaled)[i, j]


def _pairwise_same_group(ids: list[str]) -> Tensor:
    """
    Return a boolean mask, ordered like :func:`_pairwise_distances`, marking same-instance pairs.

    Two samples are "same instance" (different graph realisations of one theta, i.e. replicas)
    when they share ``instance_id`` -- the exact grouping key
    :meth:`~dcba.wrappers.supcon.DCBASupConWrapper._instance_labels` hashes to build positive
    pairs during training.

    :param ids: ``instance_id`` for each of the ``N`` samples, in the same order as the tensors
        passed to :func:`_pairwise_distances`.

    :returns: ``(N * (N - 1) / 2,)`` bool tensor.
    """
    _, group = np.unique(ids, return_inverse=True)
    group_t = torch.from_numpy(group)
    i, j = torch.triu_indices(len(ids), len(ids), offset=1)
    return group_t[i] == group_t[j]


def _pearson(x: Tensor, y: Tensor) -> float:
    """Return the Pearson correlation coefficient between two 1-D tensors."""
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def _standardize_2d(coords: Tensor) -> Tensor:
    """
    Zero-mean, unit-variance standardise 2D coordinates.

    Both t-SNE and UMAP output an arbitrary scale/orientation that differs per fit (per
    embedding, per run) even with a fixed seed, since it depends on the input data itself --
    without this, plots that happen to look "more spread out" may just be an artefact of that
    fit's arbitrary scale rather than the embedding actually being less clustered. Standardising
    puts every projection in this script on the same units, so :data:`PROJ_AXIS_LIM` can apply
    uniformly and cluster spread is comparable across plots.

    :param coords: ``(N, 2)`` raw projected coordinates.

    :returns: ``(N, 2)`` standardised coordinates.
    """
    return (coords - coords.mean(dim=0)) / coords.std(dim=0).clamp(min=1e-8)


def _project_tsne(z: Tensor) -> Tensor:
    """
    Project ``z`` to 2D with t-SNE, standardised (see :func:`_standardize_2d`).

    Non-linear, since the relationship between theta distance and embedding distance is not
    assumed to be linear (unlike PCA, which would flatten away any such structure).

    :param z: ``(N, D)`` unit-norm embedding tensor.

    :returns: ``(N, 2)`` standardised projected coordinates.
    """
    perplexity = min(30, max(1, z.size(0) - 1))
    tsne = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity, init="pca", random_state=SEED
    )
    return _standardize_2d(torch.from_numpy(tsne.fit_transform(z.numpy())))


def _project_umap(z: Tensor) -> Tensor:
    """
    Project ``z`` to 2D with UMAP, standardised (see :func:`_standardize_2d`).

    A second non-linear projection alongside t-SNE (:func:`_project_tsne`) for comparison --
    UMAP better preserves global/large-scale structure, where t-SNE emphasises local
    neighbourhoods, so the two can disagree on cluster layout even for the same embedding.

    :param z: ``(N, D)`` unit-norm embedding tensor.

    :returns: ``(N, 2)`` standardised projected coordinates.
    """
    n_neighbors = min(15, max(2, z.size(0) - 1))
    reducer = umap.UMAP(n_components=2, metric="cosine", n_neighbors=n_neighbors, random_state=SEED)
    return _standardize_2d(torch.from_numpy(reducer.fit_transform(z.numpy())))


def _plot_distance_scatter(
    d_theta: Tensor,
    d_embed: Tensor,
    same_group: Tensor,
    embed_name: str,
    out_path: Path,
    n_bins: int = 20,
    band: tuple[float, float] = (0.25, 0.75),
    n_outlier_sample: int = 3000,
    show_same_instance: bool = True,
) -> None:
    """
    Plot pairwise ``d_theta`` against embedding distance as a binned mean trend with a
    percentile band, plus a same-instance reference point.

    Rendering all ``O(N^2)`` pairs directly (tens of millions on a full test split) -- as a raw
    scatter, a hexbin, or a smoothed density map -- either overplots into a solid blob or, when
    density is spread broadly across the whole y-range (as it is for a collapsed embedding), just
    reads as a dense wall with no visible structure. A mean line with a spread band conveys the
    same "does distance scale with d_theta" question directly, independent of how the underlying
    joint density happens to be shaped.

    Same-instance pairs (see ``same_group``) are excluded from the trend/band: they all sit at
    ``d_theta ~ 0`` by construction (same theta), so mixed into equal-count bins alongside
    genuinely-different, merely-similar instances they only dilute the leftmost bin's x-position
    away from 0. They are shown separately, as a single point at their own (near-zero) d_theta
    with a percentile error bar, rather than a full-width reference line that would wrongly
    suggest they are relevant at every d_theta.

    A light random subsample of the (same-instance-excluded) raw pairs is also drawn underneath,
    so genuine outliers and the true low-d_theta extent are visible beyond what the smoothed
    band/bins alone would show.

    :param d_theta: Pairwise theta distances.
    :param d_embed: Pairwise embedding distances, same pairing/order as ``d_theta``.
    :param same_group: Boolean mask, same pairing/order as ``d_theta``, marking pairs that are
        different graph realisations of the same instance (see :func:`_pairwise_same_group`).
    :param embed_name: Embedding name for axis/title labels (e.g. ``"z_g"``).
    :param out_path: PNG output path.
    :param n_bins: Number of equal-count bins used for the trend line and band.
    :param band: ``(lo, hi)`` quantiles (in ``[0, 1]``) shown as the shaded spread band.
    :param n_outlier_sample: Max number of cross-instance pairs drawn as light context points.
    :param show_same_instance: Whether to draw the same-instance reference point at all -- set
        ``False`` for a version of the plot with only the cross-instance trend, unobstructed.
    """
    cross = ~same_group
    d_theta_cross = d_theta[cross]
    d_embed_cross = d_embed[cross]

    order = torch.argsort(d_theta_cross)
    d_theta_sorted = d_theta_cross[order]
    d_embed_sorted = d_embed_cross[order]
    bin_edges = torch.linspace(0, len(d_theta_sorted), n_bins + 1).long()
    band_t = torch.tensor(band)
    xs, means, lo, hi = [], [], [], []
    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        if end <= start:
            continue
        chunk = d_embed_sorted[start:end]
        xs.append(d_theta_sorted[start:end].mean().item())
        means.append(chunk.mean().item())
        q_lo, q_hi = torch.quantile(chunk, band_t)
        lo.append(q_lo.item())
        hi.append(q_hi.item())

    embed_math = EMBED_MATH[embed_name]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    n_cross = d_theta_cross.size(0)
    if n_cross > 0:
        sample_idx = torch.randperm(n_cross, generator=torch.Generator().manual_seed(SEED))[
            : min(n_outlier_sample, n_cross)
        ]
        ax.scatter(
            d_theta_cross[sample_idx].numpy(),
            d_embed_cross[sample_idx].numpy(),
            s=4,
            alpha=0.15,
            color="steelblue",
            linewidths=0,
            zorder=1,
            label=f"raw pairs (n={sample_idx.numel()} sampled of {n_cross})",
        )

    ax.fill_between(
        xs,
        lo,
        hi,
        color="steelblue",
        alpha=0.3,
        zorder=2,
        label=f"{int(band[0] * 100)}-{int(band[1] * 100)}th percentile",
    )
    ax.plot(xs, means, color="firebrick", linewidth=2.5, zorder=3, label="binned mean")

    if show_same_instance and same_group.any():
        same_d_theta = d_theta[same_group].mean().item()
        same_d_embed = d_embed[same_group]
        same_mean = same_d_embed.mean().item()
        same_lo, same_hi = torch.quantile(same_d_embed, band_t)
        ax.errorbar(
            [same_d_theta],
            [same_mean],
            yerr=[[same_mean - same_lo.item()], [same_hi.item() - same_mean]],
            fmt="o",
            markerfacecolor="none",
            markeredgecolor="dimgray",
            ecolor="dimgray",
            alpha=0.7,
            markersize=6,
            markeredgewidth=1.3,
            capsize=3,
            linewidth=1.1,
            zorder=4,
            label=f"same instance (n={same_group.sum().item()})",
        )

    ax.set_xlabel(r"$d_\theta$ (scaled ABCD parameter distance)")
    ax.set_ylabel(rf"$d({embed_math})$")
    ax.set_ylim(0, 2)
    ax.set_title(rf"Distance preservation -- ${embed_math}$")
    sns.despine(ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _epanechnikov_density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Estimate a 1D density over ``grid`` from samples ``x`` with an Epanechnikov-kernel KDE.

    Note this kernel has *compact* support: density is exactly zero more than one bandwidth from
    every sample, unlike a Gaussian kernel's smooth infinite tail. A handful of genuine outliers
    sitting further than one bandwidth from the rest of their group will still show as a
    disconnected "island" of nonzero density rather than a continuous tail -- that is a real
    (if rare) subset of the data, not a KDE artefact, but it can look surprising on a log-scale
    axis where the zero in between renders as a gap down to the axis floor.

    :param x: ``(N,)`` samples.
    :param grid: ``(M,)`` points to evaluate the density at.

    :returns: ``(M,)`` density values.
    """
    std = x.std() if x.size > 1 else 1.0
    # Silverman's rule of thumb (1.06 * std * n^-1/5) assumes a Gaussian kernel; Epanechnikov
    # needs roughly 2.2x that bandwidth for equally smooth output, or it under-smooths.
    bandwidth = max(2.2 * 1.06 * std * x.size ** (-1 / 5), 1e-3)
    kde = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(x.reshape(-1, 1))
    return np.exp(kde.score_samples(grid.reshape(-1, 1)))


def _stability_densities(
    d_embed: Tensor, same_group: Tensor, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(same_instance_density, different_instance_density)`` over ``grid``."""
    return (
        _epanechnikov_density(d_embed[same_group].numpy(), grid),
        _epanechnikov_density(d_embed[~same_group].numpy(), grid),
    )


def _stability_ylim(density_pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """
    Shared log-scale y-limits across multiple stability plots.

    z_g and z_theta can have very different peak heights (e.g. a collapsed embedding's
    same-instance spike towers over its different-instance density); a log y-axis on a shared
    range makes both curves visible on every plot and lets peak heights be compared directly
    across embeddings, rather than each plot auto-scaling to its own data.

    :param density_pairs: One ``(same, different)`` density array pair per plot that will share
        this y-axis.

    :returns: ``(ymin, ymax)`` for :meth:`matplotlib.axes.Axes.set_ylim` on a log-scale axis.
    """
    all_densities = np.concatenate([d for pair in density_pairs for d in pair])
    positive = all_densities[all_densities > 1e-6]
    ymin = positive.min() if positive.size else 1e-3
    ymax = all_densities.max() * 1.3
    return ymin, ymax


def _plot_stability_distribution(
    d_embed: Tensor,
    same_group: Tensor,
    embed_name: str,
    out_path: Path,
    ylim: tuple[float, float],
) -> None:
    """
    Compare embedding distance for same-instance pairs against all other pairs.

    Different graph realisations of the same theta ("replicas") should embed close together if
    the encoder is stable to the graph-sampling noise -- this plots the two pairwise-distance
    distributions overlaid so that expectation is directly checkable, rather than reading it off
    a single reference line on the distance-preservation plot (:func:`_plot_distance_scatter`).

    :param d_embed: Pairwise embedding distances.
    :param same_group: Boolean mask, same pairing/order as ``d_embed``, marking same-instance
        pairs (see :func:`_pairwise_same_group`).
    :param embed_name: Embedding name for axis/title labels (e.g. ``"z_g"``).
    :param out_path: PNG output path.
    :param ylim: Shared log-scale y-limits (see :func:`_stability_ylim`) so this plot and its
        counterpart for the other embedding are on the same axis and directly comparable.
    """
    embed_math = EMBED_MATH[embed_name]
    grid = np.linspace(0, 2, 400)
    same_density, diff_density = _stability_densities(d_embed, same_group, grid)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for density, mask, color, group_name in [
        (same_density, same_group, "firebrick", "same instance"),
        (diff_density, ~same_group, "steelblue", "different instance"),
    ]:
        ax.fill_between(grid, density, color=color, alpha=0.4)
        ax.plot(
            grid, density, color=color, linewidth=1.8, label=f"{group_name} (n={mask.sum().item()})"
        )
    ax.set_xlabel(rf"$d({embed_math})$")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_title(rf"Embedding stability -- ${embed_math}$")
    sns.despine(ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_projection(
    coords: Tensor, values: Tensor, key: str, embed_name: str, method: str, out_path: Path
) -> None:
    """
    Plot a standardised 2D projection of an embedding, coloured by one raw ABCD parameter.

    :param coords: ``(N, 2)`` standardised projected coordinates (see :func:`_project_tsne` /
        :func:`_project_umap`).
    :param values: ``(N,)`` raw parameter values used for point colour.
    :param key: Parameter name, used for the colour bar label.
    :param embed_name: Embedding name for the title (e.g. ``"z_g"``).
    :param method: Projection method name for axis/title labels (e.g. ``"t-SNE"``, ``"UMAP"``).
    :param out_path: PNG output path.
    """
    values_np = values.numpy()
    norm = plt.Normalize(values_np.min(), values_np.max())
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    mappable.set_array(values_np)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        x=coords[:, 0].numpy(),
        y=coords[:, 1].numpy(),
        hue=values_np,
        palette="viridis",
        s=40,
        linewidth=0,
        legend=False,
        ax=ax,
    )
    fig.colorbar(mappable, ax=ax, label=key)
    ax.set_title(rf"$({EMBED_MATH[embed_name]})$ {method} projection, coloured by {key}")
    ax.set_xlabel(f"{method} 1 (standardised)")
    ax.set_ylabel(f"{method} 2 (standardised)")
    ax.set_xlim(-PROJ_AXIS_LIM, PROJ_AXIS_LIM)
    ax.set_ylim(-PROJ_AXIS_LIM, PROJ_AXIS_LIM)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_embeddings(emb: RunEmbeddings, out_path: Path) -> None:
    """
    Save the collected per-graph embeddings and their matching keys to a ``.pt`` file.

    Persists ``z_g`` (and ``z_theta``) row-aligned with ``instance_id``/``replica`` and the theta
    tensors, so a downstream consumer can load the graph embeddings and match each row back to its
    exact source graph. This is written for later Mahalanobis distance learning over ``z_g`` (which
    needs the raw per-graph embeddings plus their instance grouping), not used by this script's own
    plots. Row ``i`` across every field refers to the same graph; ``(instance_id[i], replica[i])``
    is its unique key.

    :param emb: Collected embeddings for the run.
    :param out_path: Output ``.pt`` file path; parent directory is created if missing.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": emb.run_id,
            "run_name": emb.run_name,
            "z_g": emb.z_g,
            "z_theta": emb.z_theta,
            "theta_scaled": emb.theta_scaled,
            "theta_raw": emb.theta_raw,
            "instance_id": emb.instance_id,
            "replica": emb.replica,
        },
        out_path,
    )
    logger.info(f"Saved {emb.z_g.size(0)} embeddings to {out_path}")


def _analyse_run(emb: RunEmbeddings, run_dir: Path) -> tuple[float, float]:
    """
    Produce all plots for one run's collected embeddings and return its correlations.

    :param emb: Collected theta/embedding tensors for the run.
    :param run_dir: Output directory for this run's PNG files; created if missing.

    :returns: ``(corr_zg, corr_ztheta)`` -- Pearson correlation between ``d_theta`` and each
        embedding's pairwise distance.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    d_theta = _pairwise_config_distance(emb.theta_scaled)
    d_zg = _pairwise_distances(emb.z_g)
    d_ztheta = _pairwise_distances(emb.z_theta)
    same_group = _pairwise_same_group(emb.instance_id)

    _plot_distance_scatter(d_theta, d_zg, same_group, "z_g", run_dir / "dtheta_vs_dzg.png")
    _plot_distance_scatter(
        d_theta, d_ztheta, same_group, "z_theta", run_dir / "dtheta_vs_dztheta.png"
    )
    _plot_distance_scatter(
        d_theta,
        d_zg,
        same_group,
        "z_g",
        run_dir / "dtheta_vs_dzg_no_same_instance.png",
        show_same_instance=False,
    )
    _plot_distance_scatter(
        d_theta,
        d_ztheta,
        same_group,
        "z_theta",
        run_dir / "dtheta_vs_dztheta_no_same_instance.png",
        show_same_instance=False,
    )
    stability_grid = np.linspace(0, 2, 400)
    zg_densities = _stability_densities(d_zg, same_group, stability_grid)
    ztheta_densities = _stability_densities(d_ztheta, same_group, stability_grid)
    stability_ylim = _stability_ylim([zg_densities, ztheta_densities])
    _plot_stability_distribution(
        d_zg, same_group, "z_g", run_dir / "stability_zg.png", ylim=stability_ylim
    )
    _plot_stability_distribution(
        d_ztheta, same_group, "z_theta", run_dir / "stability_ztheta.png", ylim=stability_ylim
    )

    projections = {
        "t-SNE": (_project_tsne(emb.z_g), _project_tsne(emb.z_theta)),
        "UMAP": (_project_umap(emb.z_g), _project_umap(emb.z_theta)),
    }
    for method, (coords_zg, coords_ztheta) in projections.items():
        method_slug = method.lower().replace("-", "")
        for i, key in enumerate(ABCD_CONFIG_KEYS):
            values = emb.theta_raw[:, i]
            _plot_projection(
                coords_zg,
                values,
                key,
                "z_g",
                method,
                run_dir / f"proj_{method_slug}_zg_by_{key}.png",
            )
            _plot_projection(
                coords_ztheta,
                values,
                key,
                "z_theta",
                method,
                run_dir / f"proj_{method_slug}_ztheta_by_{key}.png",
            )

    return _pearson(d_theta, d_zg), _pearson(d_theta, d_ztheta)


def main() -> None:
    """Run the embedding stability analysis for each run in :data:`RUNS`."""
    device = torch.device(DEVICE)
    api = wandb.Api()

    logger.info(
        f"{'run_id':<10} {'n':>5} {'corr(d_theta, d_zg)':>20} {'corr(d_theta, d_ztheta)':>24}  label"
    )
    for run_id, run_name in RUNS:
        logger.info(f"Processing run {run_id} ({run_name})...")
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        ckpt_path = _download_checkpoint(run, CACHE_DIR)
        cfg = _apply_config_override(run.config, run_id)
        wrapper, scaler = _load_wrapper(cfg, ckpt_path, device)
        emb = _collect_embeddings(
            wrapper,
            scaler,
            cfg,
            run_id=run_id,
            run_name=run_name,
            num_instances=NUM_INSTANCES,
            device=device,
        )
        run_dir = OUTPUT_DIR / run_id
        _save_embeddings(emb, run_dir / "embeddings.pt")
        corr_zg, corr_ztheta = _analyse_run(emb, run_dir)
        logger.info(
            f"{run_id:<10} {emb.theta_raw.size(0):>5} {corr_zg:>20.3f} {corr_ztheta:>24.3f}  {run_name}"
        )
        logger.info(f"Wrote plots to {run_dir}")


if __name__ == "__main__":
    main()
