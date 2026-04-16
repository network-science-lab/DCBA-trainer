"""Lightning logger factory."""

import logging
from unittest.mock import MagicMock

from lightning.pytorch import loggers

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
    try:
        return loggers.WandbLogger(
            project=config["training"]["logger"]["project"],
            name=config["training"]["logger"].get("name"),
            tags=config["training"]["logger"].get("tags", []),
            save_dir=".wandb",
            offline=config["training"]["logger"].get("offline", False),
        )
    except Exception as exc:
        logger.warning("WandbLogger not initialised — using dummy. Reason: %s", exc)
        return DummyLogger()
