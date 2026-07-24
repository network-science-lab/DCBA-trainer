"""Lightning logger factory."""

import logging
from unittest.mock import MagicMock

import wandb
from lightning.pytorch import loggers
from omegaconf import OmegaConf

from dcba.utils.misc import unflatten_dict

logger = logging.getLogger(__name__)


class DummyLogger(MagicMock):
    """No-op logger used when WandbLogger cannot be initialised."""


def get_logger(config: dict) -> loggers.WandbLogger | DummyLogger:
    """
    Return a WandbLogger configured from the training config, or a no-op dummy on failure.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).

    :returns: A :class:`~lightning.pytorch.loggers.WandbLogger` if wandb is reachable,
        otherwise a :class:`DummyLogger` that silently absorbs all calls.
    """
    logger_cfg = config["training"]["logger"]
    try:
        run = (
            wandb.init(
                project=logger_cfg["project"],
                name=logger_cfg.get("name"),
                tags=logger_cfg.get("tags", []),
                # Resume an existing run when its id is supplied (e.g. after a crash); otherwise
                # wandb mints a fresh run. ``resume="allow"`` continues that id if it exists and
                # starts it if it does not.
                id=logger_cfg.get("id"),
                resume=logger_cfg.get("resume", "allow" if logger_cfg.get("id") else None),
            )
            if wandb.run is None
            else wandb.run
        )
        sweep_cfg = unflatten_dict(dict(run.config))
        if sweep_cfg:
            config = OmegaConf.merge(config, OmegaConf.create(sweep_cfg))

        return loggers.WandbLogger(
            experiment=run,
            project=config["training"]["logger"]["project"],
            name=config["training"]["logger"].get("name"),
            tags=config["training"]["logger"].get("tags", []),
            save_dir=".wandb",
            offline=config["training"]["logger"].get("offline", False),
        )
    except Exception as exc:
        logger.warning("WandbLogger not initialised — using dummy. Reason: %s", exc)
        return DummyLogger()
