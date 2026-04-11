"""Training entry point — orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl

from dcba.datamodule import ConfigDataModule
from dcba.models.config_encoder import ConfigEncoder
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.wrapper import ConfigAutoencoderWrapper

_WRAPPERS = {
    "config_autoencoder": ConfigAutoencoderWrapper,
}


def train(config: dict) -> None:
    """
    Run a full training and test cycle.

    Instantiates the datamodule, model, and wrapper named in
    ``config["training"]["wrapper"]``, then calls ``trainer.fit()`` followed by
    ``trainer.test()``.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).
    """
    data_cfg = config["data"]
    datamodule = ConfigDataModule(
        report_path=Path(data_cfg["report_path"]),
        n_max=data_cfg["n_max"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
    )

    model_cfg = config["model"]
    encoder = ConfigEncoder(
        input_dim=model_cfg["input_dim"],
        hidden_dims=list(model_cfg["hidden_dims"]),
        embedding_dim=model_cfg["embedding_dim"],
    )

    wrapper_name = config["training"]["wrapper"]
    if wrapper_name not in _WRAPPERS:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")
    wrapper = _WRAPPERS[wrapper_name](encoder, config["training"]["optimizer"]["args"])

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=get_logger(config),
    )
    trainer.fit(wrapper, datamodule=datamodule)
    wrapper._scaler = datamodule.scaler
    trainer.test(wrapper, datamodule=datamodule)
