# ruff: noqa
"""Embedding stability analysis for gps-{ae,vae}-supcon runs logged to Weights & Biases.

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
- ``proj2d_{zg,ztheta}_by_<key>.png``: 2D t-SNE projection of the embedding, coloured by each raw
  ABCD parameter, to check whether embeddings cluster along that parameter.

Prerequisites:

- ``wandb login`` (or an existing ``.netrc`` entry) with read access to the run's project.
- The ABCD dataset directory recorded in each run's config (``data.dataset_root``) must already
  exist and be readable on the machine this script runs on -- pull it with ``dvc pull`` first if
  needed. This script does not fetch data itself.

Usage:
    uv run scripts/embedding_stability.py

To analyse different runs, edit :data:`RUNS` at the top of this file.
"""

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from dcba.datamodule import ABCDDataModule
from dcba.dataset import ABCDConfigScaler
from dcba.dataset.transforms import ABCD_CONFIG_KEYS
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

# Paper-ready styling: figures are saved at PLOT_DPI regardless of on-screen figsize.
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

#: (run_id, human-readable label) for the current gps-ae/vae-supcon comparison set.
RUNS: list[tuple[str, str]] = [
    ("3or3wpqk", "gps-ae-supcon, no-community-features, batch64"),
    ("oraxoj6n", "gps-ae-supcon, no-community-features, batch64"),
]


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
    graph realisations of the same theta ("replicas").
    """

    run_id: str
    run_name: str
    theta_raw: Tensor
    theta_scaled: Tensor
    z_g: Tensor
    z_theta: Tensor
    instance_id: list[str]


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
            "(gps-ae/vae-supcon) is handled by this script."
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
    for batch in tqdm(loader, desc=f"{run_id} ({run_name})", unit="batch"):
        batch = batch.to(device)
        theta = wrapper._unpack_batch(batch)
        z_g, z_theta = wrapper.encode(batch)
        theta_chunks.append(theta.cpu())
        z_g_chunks.append(z_g.cpu())
        z_theta_chunks.append(z_theta.cpu())
        instance_ids.extend(batch.instance_id)

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


def _project_2d(z: Tensor) -> Tensor:
    """
    Project ``z`` to 2D with t-SNE.

    Non-linear, since the relationship between theta distance and embedding distance is not
    assumed to be linear (unlike PCA, which would flatten away any such structure).

    :param z: ``(N, D)`` unit-norm embedding tensor.

    :returns: ``(N, 2)`` projected coordinates.
    """
    perplexity = min(30, max(1, z.size(0) - 1))
    tsne = TSNE(
        n_components=2, metric="cosine", perplexity=perplexity, init="pca", random_state=SEED
    )
    return torch.from_numpy(tsne.fit_transform(z.numpy()))


def _plot_distance_scatter(
    d_theta: Tensor,
    d_embed: Tensor,
    same_group: Tensor,
    embed_name: str,
    out_path: Path,
    n_bins: int = 20,
    band: tuple[float, float] = (0.25, 0.75),
) -> None:
    """
    Plot pairwise ``d_theta`` against embedding distance as a binned mean trend with a
    percentile band, plus a same-instance noise-floor reference line.

    Rendering all ``O(N^2)`` pairs directly (tens of millions on a full test split) -- as a raw
    scatter, a hexbin, or a smoothed density map -- either overplots into a solid blob or, when
    density is spread broadly across the whole y-range (as it is for a collapsed embedding), just
    reads as a dense wall with no visible structure. A mean line with a spread band conveys the
    same "does distance scale with d_theta" question directly, independent of how the underlying
    joint density happens to be shaped.

    :param d_theta: Pairwise theta distances.
    :param d_embed: Pairwise embedding distances, same pairing/order as ``d_theta``.
    :param same_group: Boolean mask, same pairing/order as ``d_theta``, marking pairs that are
        different graph realisations of the same instance (see :func:`_pairwise_same_group`).
        Their embedding distance is pure realisation noise, not signal, so it is called out as a
        separate horizontal reference line.
    :param embed_name: Embedding name for axis/title labels (e.g. ``"z_g"``).
    :param out_path: PNG output path.
    :param n_bins: Number of equal-count bins used for the trend line and band.
    :param band: ``(lo, hi)`` quantiles (in ``[0, 1]``) shown as the shaded spread band.
    """
    order = torch.argsort(d_theta)
    d_theta_sorted = d_theta[order]
    d_embed_sorted = d_embed[order]
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
    ax.fill_between(
        xs,
        lo,
        hi,
        color="steelblue",
        alpha=0.3,
        label=f"{int(band[0] * 100)}-{int(band[1] * 100)}th percentile",
    )
    ax.plot(xs, means, color="firebrick", linewidth=2.5, label="binned mean")

    if same_group.any():
        floor_mean = d_embed[same_group].mean().item()
        ax.axhline(
            floor_mean,
            color="dimgray",
            linestyle="--",
            linewidth=1.5,
            label=f"same-instance noise floor (n={same_group.sum().item()})",
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

    Bandwidth is set via Silverman's rule of thumb -- the same heuristic used for the default
    Gaussian KDE -- since :class:`~sklearn.neighbors.KernelDensity` requires an explicit
    bandwidth for a non-Gaussian kernel.

    :param x: ``(N,)`` samples.
    :param grid: ``(M,)`` points to evaluate the density at.

    :returns: ``(M,)`` density values.
    """
    std = x.std() if x.size > 1 else 1.0
    bandwidth = max(1.06 * std * x.size ** (-1 / 5), 1e-3)
    kde = KernelDensity(kernel="epanechnikov", bandwidth=bandwidth).fit(x.reshape(-1, 1))
    return np.exp(kde.score_samples(grid.reshape(-1, 1)))


def _plot_stability_distribution(
    d_embed: Tensor, same_group: Tensor, embed_name: str, out_path: Path
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
    """
    embed_math = EMBED_MATH[embed_name]
    grid = np.linspace(0, 2, 400)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    for mask, color, group_name in [
        (same_group, "firebrick", "same instance"),
        (~same_group, "steelblue", "different instance"),
    ]:
        density = _epanechnikov_density(d_embed[mask].numpy(), grid)
        ax.fill_between(grid, density, color=color, alpha=0.4)
        ax.plot(
            grid, density, color=color, linewidth=1.8, label=f"{group_name} (n={mask.sum().item()})"
        )
    ax.set_xlabel(rf"$d({embed_math})$")
    ax.set_ylabel("Density")
    ax.set_title(rf"Embedding stability -- ${embed_math}$")
    sns.despine(ax=ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_projection(
    coords: Tensor, values: Tensor, key: str, embed_name: str, out_path: Path
) -> None:
    """
    Plot a 2D t-SNE projection of an embedding, coloured by one raw ABCD parameter.

    :param coords: ``(N, 2)`` t-SNE-projected coordinates (see :func:`_project_2d`).
    :param values: ``(N,)`` raw parameter values used for point colour.
    :param key: Parameter name, used for the colour bar label.
    :param embed_name: Embedding name for the title (e.g. ``"z_g"``).
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
    ax.set_title(rf"$({EMBED_MATH[embed_name]})$ t-SNE projection, coloured by {key}")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


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
    _plot_stability_distribution(d_zg, same_group, "z_g", run_dir / "stability_zg.png")
    _plot_stability_distribution(d_ztheta, same_group, "z_theta", run_dir / "stability_ztheta.png")

    coords_zg = _project_2d(emb.z_g)
    coords_ztheta = _project_2d(emb.z_theta)
    for i, key in enumerate(ABCD_CONFIG_KEYS):
        values = emb.theta_raw[:, i]
        _plot_projection(coords_zg, values, key, "z_g", run_dir / f"proj2d_zg_by_{key}.png")
        _plot_projection(
            coords_ztheta, values, key, "z_theta", run_dir / f"proj2d_ztheta_by_{key}.png"
        )

    return _pearson(d_theta, d_zg), _pearson(d_theta, d_ztheta)


def main() -> None:
    """Run the embedding stability analysis for each run in :data:`RUNS`."""
    device = torch.device(DEVICE)
    api = wandb.Api()

    logger.info(f"{'run_id':<10} {'n':>5} {'corr(d_theta, d_zg)':>20} {'corr(d_theta, d_ztheta)':>24}  label")
    for run_id, run_name in RUNS:
        logger.info(f"Processing run {run_id} ({run_name})...")
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        ckpt_path = _download_checkpoint(run, CACHE_DIR)
        wrapper, scaler = _load_wrapper(run.config, ckpt_path, device)
        emb = _collect_embeddings(
            wrapper,
            scaler,
            run.config,
            run_id=run_id,
            run_name=run_name,
            num_instances=NUM_INSTANCES,
            device=device,
        )
        corr_zg, corr_ztheta = _analyse_run(emb, OUTPUT_DIR / run_id)
        logger.info(
            f"{run_id:<10} {emb.theta_raw.size(0):>5} {corr_zg:>20.3f} {corr_ztheta:>24.3f}  {run_name}"
        )
        logger.info(f"Wrote plots to {OUTPUT_DIR / run_id}")


if __name__ == "__main__":
    main()
