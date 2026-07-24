"""Training entry point -- orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn
from torch_geometric.transforms import BaseTransform

from dcba.datamodule import ABCDDataModule
from dcba.dataset import (
    ABCDConfigScaler,
    ABCDConfigSchema,
    ABCDConfigToTensor,
    ABCDIdentityConfigScaler,
    ABCDLogConfigScaler,
    ABCDNMaxConfigScaler,
    ABCDRelativeConfigScaler,
    CommunityToSize,
    ConstantNodeFeatures,
)
from dcba.models.config_ae import ConfigAutoEncoder
from dcba.models.gin_encoder import GINEncoder
from dcba.models.gps_encoder import GPSEncoder
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.training.loss import (
    ABCDConstraintPenaltyLoss,
    MultiPositiveSupConLoss,
)
from dcba.wrappers import DCBAAutoencoderWrapper, DCBASupConWrapper

_WRAPPERS = {
    "config_autoencoder": DCBAAutoencoderWrapper,
    "supcon": DCBASupConWrapper,
}

_MODELS = {
    "ConfigAutoEncoder": ConfigAutoEncoder,
    "GINEncoder": GINEncoder,
    "GPSEncoder": GPSEncoder,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "abcd_constraint": ABCDConstraintPenaltyLoss,
    "supcon": MultiPositiveSupConLoss,
}

_SCALERS: dict[str, type] = {
    "ABCDConfigScaler": ABCDConfigScaler,
    "ABCDIdentityConfigScaler": ABCDIdentityConfigScaler,
    "ABCDLogConfigScaler": ABCDLogConfigScaler,
    "ABCDNMaxConfigScaler": ABCDNMaxConfigScaler,
    "ABCDRelativeConfigScaler": ABCDRelativeConfigScaler,
}

_TRANSFORMS: dict[str, type[BaseTransform]] = {
    "ABCDConfigToTensor": ABCDConfigToTensor,
}

_NODE_TRANSFORMS: dict[str, type[BaseTransform]] = {
    "CommunityToSize": CommunityToSize,
    "ConstantNodeFeatures": ConstantNodeFeatures,
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


def _build_node_transform(name: str | None) -> BaseTransform:
    """
    Instantiate a node-feature transform from its registry name.

    :param name: Key in :data:`_NODE_TRANSFORMS`, or ``None`` to use the default
        :class:`~dcba.dataset.CommunityToSize`.

    :returns: A transform instance.
    """
    if name is None:
        return CommunityToSize()
    if name not in _NODE_TRANSFORMS:
        raise ValueError(f"Unknown node transform '{name}'. Available: {list(_NODE_TRANSFORMS)}")
    return _NODE_TRANSFORMS[name]()


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


def _attach_ordering_scaler(loss: nn.Module, config: dict) -> None:
    """
    Attach the data pipeline's scaler to a constraint loss that compares orderings in raw scale.

    No-op unless ``loss`` is an :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss` with
    ``ordering_penalties="raw"`` -- that mode denormalises predictions through the same scaler
    the datamodule uses (see the loss docstring), so the two must be built from the same config.

    :param loss: The regression loss instance built by :func:`_build_loss`.
    :param config: Full resolved training config dict.
    """
    if isinstance(loss, ABCDConstraintPenaltyLoss) and loss.ordering_mode == "raw":
        scaler = _build_scaler(config["data"].get("scaler"))
        if scaler is None:
            raise ValueError(
                "ordering_penalties='raw' requires data.scaler to be set -- the loss "
                "denormalises predictions through the same scaler the datamodule uses."
            )
        loss.set_scaler(scaler)


def build_supcon_wrapper(config: dict) -> DCBASupConWrapper:
    """
    Build an untrained :class:`~dcba.wrappers.supcon.DCBASupConWrapper` from a resolved config.

    Shared by :func:`train` and any code that needs to reconstruct a supcon wrapper's
    architecture from a logged run config independent of checkpoint weights (e.g. analysis
    scripts loading a trained checkpoint via ``load_state_dict``).

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`), with ``config["training"]["wrapper"] ==
        "supcon"``.

    :returns: An untrained :class:`~dcba.wrappers.supcon.DCBASupConWrapper`.
    """
    training_cfg = config["training"]
    losses_cfg = training_cfg["losses"]
    reg_loss = _build_loss(losses_cfg["reg"])
    _attach_ordering_scaler(reg_loss, config)
    supcon_loss = _build_loss(losses_cfg["repr"])
    graph_encoder = _build_model(config["models"]["graph"])
    config_encoder = _build_model(config["models"]["theta"])
    return DCBASupConWrapper(
        graph_encoder=graph_encoder,
        config_encoder=config_encoder,
        optimizer_config=training_cfg["optimizer"]["args"],
        reg_loss=reg_loss,
        supcon_loss=supcon_loss,
        lambda_supcon=losses_cfg["repr"].get("weight", 1.0),
        aux_weight=training_cfg.get("aux_weight", 0.1),
    )


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
    node_transform = _build_node_transform(data_cfg.get("node_transform"))
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
        node_transform=node_transform,
    )

    if wrapper_name == "config_autoencoder":
        losses_cfg = training_cfg["losses"]
        encoder = _build_model(next(iter(config["models"].values())))
        loss_fn = _build_loss(losses_cfg["reg"])
        _attach_ordering_scaler(loss_fn, config)
        wrapper: pl.LightningModule = DCBAAutoencoderWrapper(
            encoder,
            training_cfg["optimizer"]["args"],
            loss_fn=loss_fn,
        )
    elif wrapper_name == "supcon":
        wrapper = build_supcon_wrapper(config)
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
    # Optional resume: point at a Lightning checkpoint (e.g. checkpoints/last.ckpt) to restore the
    # full training state -- model weights, optimizer, epoch, global step, and callback state.
    ckpt_path = training_cfg.get("ckpt_path")
    trainer.fit(wrapper, datamodule=datamodule, ckpt_path=ckpt_path)
    wrapper.set_scaler(datamodule.scaler)
    metrics = trainer.test(wrapper, datamodule=datamodule)
    for i in Path(f"{config['hydra']['runtime']['output_dir']}/checkpoints").iterdir():
        logger.experiment.log_artifact(
            artifact_or_path=str(i),
            name=i.stem.replace("=", "-"),
            type="model",
        )
    logger.log_metrics(metrics[-1])
