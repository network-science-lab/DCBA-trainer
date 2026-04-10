"""Training entrypoint — invoke via ``uv run dcba-train``."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIGS_PATH), config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Load config and launch the training loop.

    :param cfg: Hydra-managed configuration object.
    """
    from dcba.training.trainer import train
    from dcba.utils.config import load_config

    config = load_config(cfg)
    train(config)


if __name__ == "__main__":
    main()
