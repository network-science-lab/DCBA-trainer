"""Training entry point -- orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn
from lightning.pytorch.strategies import DDPStrategy, Strategy

from dcba.datamodule import ABCDDataModule
from dcba.models.config_autoencoder import ConfigAutoEncoder
from dcba.models.gin_encoder import GINEncoder
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.training.loss import ABCDConstraintPenaltyLoss, MultiPositiveSupConLoss
from dcba.wrappers import DCBAAutoencoderWrapper, DCBASupConWrapper

_STRATEGIES: dict[str, type[Strategy]] = {
    "ddp": DDPStrategy,
}


_WRAPPERS = {
    "config_autoencoder": DCBAAutoencoderWrapper,
    "supcon": DCBASupConWrapper,
}

_MODELS = {
    "ConfigAutoEncoder": ConfigAutoEncoder,
    "GINEncoder": GINEncoder,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "abcd_constraint": ABCDConstraintPenaltyLoss,
    "supcon": MultiPositiveSupConLoss,
}


def _build_strategy(strategy_cfg: str | dict) -> str | Strategy:
    """
    Instantiate a Lightning training strategy from a config value.

    Pass a plain string (e.g. ``"auto"``) to use it as-is, or a dict with a ``name`` key and
    optional constructor kwargs (e.g. ``{name: ddp, find_unused_parameters: false}``) to
    instantiate a :class:`~lightning.pytorch.strategies.Strategy` subclass.

    ``find_unused_parameters: false`` is required when gradient checkpointing is active:
    DDP's default graph traversal incorrectly marks checkpointed parameters as
    unused, which causes an error during the first backward pass.

    :param strategy_cfg: Either a strategy name string or a dict with ``name`` and kwargs.

    :returns: A string strategy identifier or an instantiated :class:`Strategy` object.
    """
    if isinstance(strategy_cfg, str):
        return strategy_cfg
    cfg = dict(strategy_cfg)
    name = cfg.pop("name")
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_STRATEGIES)}")
    return _STRATEGIES[name](**cfg)


def _build_model(model_cfg: dict) -> nn.Module:
    """
    Instantiate a model from a config dict.

    :param model_cfg: Dict with ``cls`` (str) and constructor kwargs.

    :returns: A model instance.
    """
    model_cfg = dict(model_cfg)
    name = model_cfg.pop("cls")
    if name not in _MODELS:
        raise ValueError(f"Unknown model class '{name}'. Available: {list(_MODELS)}")
    return _MODELS[name](**model_cfg)


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

    training_cfg = config["training"]
    wrapper_name = training_cfg["wrapper"]

    data_cfg = config["data"]
    datamodule = ABCDDataModule(
        report_path=Path(data_cfg["report_path"]),
        n_max=data_cfg["n_max"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        single_replica_per_instance=wrapper_name == "config_autoencoder",
        seed=config.get("random_seed", 42),
    )

    if wrapper_name == "config_autoencoder":
        encoder = _build_model(next(iter(config["models"].values())))
        loss_fn = _build_loss(next(iter(training_cfg["losses"].values())))
        wrapper: pl.LightningModule = DCBAAutoencoderWrapper(
            encoder,
            training_cfg["optimizer"]["args"],
            loss_fn=loss_fn,
        )
    elif wrapper_name == "supcon":
        losses_cfg = training_cfg["losses"]
        reg_loss = _build_loss(losses_cfg["reg"])
        supcon_loss = _build_loss(losses_cfg["repr"])
        graph_encoder = _build_model(config["models"]["graph"])
        config_encoder = _build_model(config["models"]["theta"])
        wrapper = DCBASupConWrapper(
            graph_encoder=graph_encoder,
            config_encoder=config_encoder,
            optimizer_config=training_cfg["optimizer"]["args"],
            reg_loss=reg_loss,
            supcon_loss=supcon_loss,
            lambda_supcon=losses_cfg["repr"].get("weight", 1.0),
        )
    else:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")

    devices = training_cfg["devices"]
    n_devices = len(devices) if isinstance(devices, list) else int(devices)
    extra = {
        "effective_batch_size": config["data"]["batch_size"] * n_devices,
        "n_devices": n_devices,
    }
    logger.log_hyperparams({**{k: v for k, v in config.items() if k != "hydra"}, **extra})
    logger.watch(wrapper)
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg["accelerator"],
        devices=training_cfg["devices"],
        strategy=_build_strategy(training_cfg.get("strategy", "auto")),
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=logger,
    )
    trainer.fit(wrapper, datamodule=datamodule)
    wrapper.set_scaler(datamodule.scaler)
    metrics = trainer.test(wrapper, datamodule=datamodule)
    for i in Path(f"{config['hydra']['runtime']['output_dir']}/checkpoints").iterdir():
        logger.experiment.log_artifact(
            artifact_or_path=str(i),
            name=i.stem.replace("=", "-"),
            type="model",
        )
    logger.log_metrics(metrics[-1])
