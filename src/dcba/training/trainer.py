"""Training entry point -- orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn
from torch_geometric.transforms import BaseTransform

from dcba.datamodule import ABCDDataModule
from dcba.dataset import ABCDConfigScaler, ABCDConfigSchema, ABCDConfigToTensor
from dcba.models.config_autoencoder import ConfigAutoEncoder
from dcba.models.config_vae import ConfigVAE
from dcba.models.gin_encoder import GINEncoder
from dcba.models.gps_encoder import GPSEncoder
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.training.loss import (
    ABCDConstraintPenaltyLoss,
    KLDivergenceLoss,
    MultiPositiveSupConLoss,
)
from dcba.wrappers import DCBAAutoencoderWrapper, DCBASupConWrapper

_WRAPPERS = {
    "config_autoencoder": DCBAAutoencoderWrapper,
    "supcon": DCBASupConWrapper,
}

_MODELS = {
    "ConfigAutoEncoder": ConfigAutoEncoder,
    "ConfigVAE": ConfigVAE,
    "GINEncoder": GINEncoder,
    "GPSEncoder": GPSEncoder,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "abcd_constraint": ABCDConstraintPenaltyLoss,
    "kl_divergence": KLDivergenceLoss,
    "supcon": MultiPositiveSupConLoss,
}

_SCALERS: dict[str, type] = {
    "ABCDConfigScaler": ABCDConfigScaler,
}

_TRANSFORMS: dict[str, type[BaseTransform]] = {
    "ABCDConfigToTensor": ABCDConfigToTensor,
}

_CONFIG_SCHEMAS: dict[str, type] = {
    "ABCDConfigSchema": ABCDConfigSchema,
}


def _build_scaler(scaler_cfg: dict | None) -> ABCDConfigScaler | None:
    """
    Instantiate a scaler from a config dict, or return ``None``.

    :param scaler_cfg: Dict with ``name`` (str) and optional ``args`` (dict) keys, or ``None``
        to disable scaling.

    :returns: A scaler instance, or ``None`` when ``scaler_cfg`` is ``None``.
    """
    if scaler_cfg is None:
        return None
    name = scaler_cfg["name"]
    if name not in _SCALERS:
        raise ValueError(f"Unknown scaler '{name}'. Available: {list(_SCALERS)}")
    return _SCALERS[name](**scaler_cfg.get("args", {}))


def _build_transform(name: str | None) -> BaseTransform:
    """
    Instantiate a config transform from its registry name.

    :param name: Key in :data:`_TRANSFORMS`, or ``None`` to use the default
        :class:`~dcba.dataset.ABCDConfigToTensor`.

    :returns: A transform instance.
    """
    if name is None:
        return ABCDConfigToTensor()
    if name not in _TRANSFORMS:
        raise ValueError(f"Unknown transform '{name}'. Available: {list(_TRANSFORMS)}")
    return _TRANSFORMS[name]()


def _build_model(model_cfg: dict) -> nn.Module:
    """
    Instantiate a model from a config dict.

    If the dict contains a ``config_schema`` key its value is resolved from
    :data:`_CONFIG_SCHEMAS` (string -> class) before the constructor is called.

    :param model_cfg: Dict with ``cls`` (str) and constructor kwargs.

    :returns: A model instance.
    """
    model_cfg = dict(model_cfg)
    name = model_cfg.pop("cls")
    if "config_schema" in model_cfg:
        schema_name = model_cfg["config_schema"]
        if schema_name not in _CONFIG_SCHEMAS:
            raise ValueError(
                f"Unknown config schema '{schema_name}'. Available: {list(_CONFIG_SCHEMAS)}"
            )
        model_cfg["config_schema"] = _CONFIG_SCHEMAS[schema_name]
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
    scaler = _build_scaler(data_cfg.get("scaler"))
    transform = _build_transform(data_cfg.get("transform"))
    datamodule = ABCDDataModule(
        dataset_root=Path(data_cfg["dataset_root"]),
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        single_replica_per_instance=wrapper_name == "config_autoencoder",
        seed=config.get("random_seed", 42),
        scaler=scaler,
        transform=transform,
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
        kl_cfg = losses_cfg.get("kl")
        kl_loss = _build_loss(kl_cfg) if kl_cfg is not None else None
        wrapper = DCBASupConWrapper(
            graph_encoder=graph_encoder,
            config_encoder=config_encoder,
            optimizer_config=training_cfg["optimizer"]["args"],
            reg_loss=reg_loss,
            supcon_loss=supcon_loss,
            lambda_supcon=losses_cfg["repr"].get("weight", 1.0),
            kl_loss=kl_loss,
            beta_kl=kl_cfg.get("weight", 0.01) if kl_cfg is not None else 0.01,
        )
    else:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")

    logger.log_hyperparams({k: v for k, v in config.items() if k != "hydra"})
    logger.watch(wrapper)
    clip_val = training_cfg.get("gradient_clip_val")
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg["accelerator"],
        devices=training_cfg["devices"],
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=logger,
        gradient_clip_val=clip_val,
        gradient_clip_algorithm="norm" if clip_val is not None else None,
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
