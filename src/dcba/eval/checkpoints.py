"""Load a W&B run's best checkpoint and rebuild the trained supcon wrapper it produced."""

import copy
import re
from pathlib import Path

import torch
import wandb

from dcba.dataset import ABCDBaseConfigScaler
from dcba.training.trainer import _build_scaler, build_supcon_wrapper
from dcba.wrappers.supcon import DCBASupConWrapper

#: Per-run patches for ``models.graph`` fields that existed in the encoder class at training time
#: but were not yet logged to that run's W&B config -- without this, :func:`build_supcon_wrapper`
#: falls back to *today's* class default for the missing field, which can silently mismatch an
#: older checkpoint's shape.
#:
#: ``3or3wpqk`` logged ``num_clusters=16`` (so its checkpoint has the cluster-pooling branch) but
#: no ``cluster_size_quantiles`` -- today's default is a 5-element tuple, giving
#: ``proj_in_dim = 2 * hidden_dim + 5 = 133``, but the checkpoint's ``_proj.0.weight`` is shaped
#: for ``2 * hidden_dim + 2 = 130``. Confirmed by loading the checkpoint with ``strict=False``
#: after patching to a 2-element tuple: zero missing/unexpected keys.
CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    # "3or3wpqk": {"cluster_size_quantiles": (0.0, 1.0)},
}

#: Matches both the ``model-epoch-N`` artifact name and the ``model-epoch=N.ckpt`` file naming
#: Lightning's ``ModelCheckpoint`` filename template produces.
_EPOCH_RE = re.compile(r"model-epoch[-=](\d+)")


def apply_config_overrides(cfg: dict, run_id: str) -> dict:
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


def _epoch_number(artifact: wandb.Artifact) -> int:
    """Parse the epoch number out of a ``model-epoch-N``/``model-epoch=N`` name, else ``-1``."""
    match = _EPOCH_RE.search(artifact.name)
    return int(match.group(1)) if match else -1


def download_best_checkpoint(run: wandb.apis.public.Run, cache_dir: Path) -> Path:
    """
    Download the run's best model checkpoint artifact (``model-epoch-*``, never ``last.ckpt``).

    Lightning's ``ModelCheckpoint(save_top_k=1)`` only overwrites the checkpoint file when a new
    best ``val_loss`` is found, so normally exactly one ``model-epoch-*`` artifact is logged per
    run. A resumed run (new ``hydra.run.dir``, see the README's resume workflow) can log several,
    one per resume segment, each named after its own epoch number -- the one with the highest
    epoch number is then the most recently confirmed best and is preferred.

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
    best = max(candidates, key=_epoch_number)
    artifact_dir = Path(best.download(root=str(cache_dir / run.id)))
    ckpt_files = list(artifact_dir.glob("*.ckpt"))
    if not ckpt_files:
        raise ValueError(f"No .ckpt file found in downloaded artifact for run {run.id}.")
    return ckpt_files[0]


def load_supcon_wrapper(
    cfg: dict, ckpt_path: Path, device: torch.device
) -> tuple[DCBASupConWrapper, ABCDBaseConfigScaler | None]:
    """
    Rebuild a supcon wrapper's architecture from a run config, then load checkpoint weights.

    :param cfg: Run config as logged to W&B (mirrors the training hydra config).
    :param ckpt_path: Local path to the run's ``.ckpt`` file.
    :param device: Device to load the wrapper onto.

    :returns: The loaded wrapper (in eval mode) and the scaler used to train it.
    """
    training_cfg = cfg["training"]
    if training_cfg["wrapper"] != "supcon":
        raise ValueError(
            f"Unsupported wrapper '{training_cfg['wrapper']}' for run; only 'supcon' "
            "(gps-ae-supcon) is handled by this loader."
        )

    scaler = _build_scaler(cfg["data"].get("scaler"))
    wrapper = build_supcon_wrapper(cfg, scaler)
    state_dict = torch.load(str(ckpt_path), map_location=device)["state_dict"]
    wrapper.load_state_dict(state_dict)
    wrapper.set_scaler(scaler)
    wrapper.eval()
    wrapper.to(device)
    return wrapper, scaler
