"""Hydra config helpers."""

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def load_config(cfg: DictConfig) -> dict:
    """
    Convert a Hydra DictConfig to a plain dict and attach Hydra runtime metadata.

    :param cfg: The Hydra-managed config object passed to the ``@hydra.main`` function.

    :returns: A plain Python dict with all values resolved, plus a ``"hydra"`` key
        containing the Hydra runtime configuration.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    config["hydra"] = OmegaConf.to_container(HydraConfig.get(), resolve=False)
    return config
