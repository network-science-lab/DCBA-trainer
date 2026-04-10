"""Lightning logger factory."""

import logging

from lightning.pytorch import loggers

logger = logging.getLogger(__name__)


class _DummyLogger:
    """No-op logger used when WandbLogger cannot be initialised."""

    def __getattr__(self, name: str):
        """Return a no-op callable for any attribute access."""
        return lambda *a, **kw: None


def get_logger(config: dict) -> loggers.WandbLogger | _DummyLogger:
    """
    Return a WandbLogger configured from the training config, or a no-op dummy on failure.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).

    :returns: A :class:`~lightning.pytorch.loggers.WandbLogger` if wandb is reachable,
        otherwise a :class:`_DummyLogger` that silently absorbs all calls.
    """
    try:
        return loggers.WandbLogger(
            project=config["training"]["logger"]["project"],
            name=config["training"]["logger"].get("name"),
            tags=config["training"]["logger"].get("tags", []),
            save_dir=config["hydra"]["run"]["dir"],
        )
    except Exception as exc:
        logger.warning("WandbLogger not initialised — using dummy. Reason: %s", exc)
        return _DummyLogger()
