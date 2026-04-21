"""Training entry point — orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn

from dcba.datamodule import ABCDDataModule
from dcba.models.config_encoder import ConfigEncoder
from dcba.models.graph_encoder import GraphEncoder
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.training.loss import ABCDConstraintPenaltyLoss
from dcba.wrapper import DCBAAutoencoderWrapper

_WRAPPERS = {
    "config_autoencoder": DCBAAutoencoderWrapper,
}

_MODELS = {
    "ConfigEncoder": ConfigEncoder,
    "GraphEncoder": GraphEncoder,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "abcd_constraint": ABCDConstraintPenaltyLoss,
}


def _build_loss(loss_cfg: dict) -> nn.Module:
    """
    Instantiate a loss module from a config dict.

    :param loss_cfg: Dict with ``name`` (str) and ``args`` (dict) keys.

    :returns: A loss module instance.
    """
    name = loss_cfg["name"]
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSSES)}")
    return _LOSSES[name](**loss_cfg.get("args", {}))


def train(config: dict) -> None:
    """
    Run a full training and test cycle.

    Instantiates the datamodule, model, and wrapper named in
    ``config["training"]["wrapper"]``, then calls ``trainer.fit()`` followed by
    ``trainer.test()``.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).
    """
    logger = get_logger(config)

    model_cfg = config["model"]
    model_name = config["training"]["model_cls"]
    if model_name not in _MODELS:
        raise ValueError(f"Unknown model class '{model_name}'. Available: {list(_MODELS)}")
    model_cls = _MODELS[model_name]
    encoder = model_cls(**model_cfg)

    data_cfg = config["data"]
    datamodule = ABCDDataModule(
        report_path=Path(data_cfg["report_path"]),
        n_max=data_cfg["n_max"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        single_replica_per_instance=True if model_name == "ConfigEncoder" else False,
        seed=config.get("random_seed", 42),
    )

    loss_fn = _build_loss(config["training"]["loss"])

    wrapper_name = config["training"]["wrapper"]
    if wrapper_name not in _WRAPPERS:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")
    wrapper = _WRAPPERS[wrapper_name](
        encoder,
        config["training"]["optimizer"]["args"],
        loss_fn=loss_fn,
    )

    logger.log_hyperparams({key: value for key, value in config.items() if key != "hydra"})
    logger.watch(wrapper)
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=logger,
    )
    trainer.fit(wrapper, datamodule=datamodule)
    wrapper._scaler = datamodule.scaler
    metrics = trainer.test(wrapper, datamodule=datamodule)
    for i in Path(f"{config['hydra']['runtime']['output_dir']}/checkpoints").iterdir():
        logger.experiment.log_artifact(
            artifact_or_path=str(i),
            name=i.stem.replace("=", "-"),
            type="model",
        )
    logger.log_metrics(metrics[-1])
