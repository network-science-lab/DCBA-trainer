"""Hydra config helpers."""

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from dcba.utils.paths import DATA_ROOT


def load_config(cfg: DictConfig) -> dict:
    """
    Convert a Hydra DictConfig to a plain dict and attach Hydra runtime metadata.

    ``data.dataset_root`` is treated as a path relative to ``DATA_ROOT`` (i.e. the
    ``DCBA_DATA_ROOT`` environment variable) and is resolved to an absolute path here.

    :param cfg: The Hydra-managed config object passed to the ``@hydra.main`` function.

    :returns: A plain Python dict with all values resolved, plus a ``"hydra"`` key
        containing the Hydra runtime configuration.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    config["hydra"] = OmegaConf.to_container(HydraConfig.get(), resolve=False)
    config["data"]["dataset_root"] = str(DATA_ROOT / config["data"]["dataset_root"])
    return config
