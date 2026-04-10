"""Training entry point — orchestrates Lightning Trainer, datamodule, and wrapper."""

import lightning.pytorch as pl

from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger


def train(config: dict) -> None:
    """
    Run a full training and test cycle.

    Model instantiation is deferred until :mod:`dcba.wrapper` is implemented.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).
    """
    raise NotImplementedError  # fill in once wrapper.py is implemented

    trainer = pl.Trainer(  # noqa: F841
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=get_logger(config),
    )
    # trainer.fit(wrapper, datamodule=datamodule)
    # trainer.test(wrapper, datamodule=datamodule)
