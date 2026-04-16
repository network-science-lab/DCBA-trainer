"""CLI entrypoint for training — invoke via ``uv run dcba-train``."""

from pathlib import Path

import hydra
import wandb
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

load_dotenv()  # populate os.environ from .env if present; no-op otherwise

_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIGS_PATH), config_name="base")
def main(cfg: DictConfig) -> None:
    """
    Load config and launch the training loop.

    :param cfg: Hydra-managed configuration object.
    """
    from dcba.training.trainer import train
    from dcba.utils.config import load_config
    from dcba.utils.misc import unflatten_dict

    config = load_config(cfg)

    run = wandb.init()
    sweep_cfg = unflatten_dict(dict(run.config))
    if sweep_cfg:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(sweep_cfg))

    train(config)


if __name__ == "__main__":
    main()
