"""Training entry point -- orchestrates Lightning Trainer, datamodule, and wrapper."""

from pathlib import Path

import lightning.pytorch as pl
import torch.nn as nn

from dcba.datamodule import ABCDDataModule
from dcba.models.config_autoencoder import ConfigAutoEncoder
from dcba.models.ff_graph_config_predictor import FeedforwardGraphConfigPredictor
from dcba.training.callbacks import get_callbacks
from dcba.training.loggers import get_logger
from dcba.training.loss import ABCDConstraintPenaltyLoss, MultiPositiveSupConLoss
from dcba.wrappers import DCBAAutoencoderWrapper, DCBASupConWrapper

_WRAPPERS = {
    "config_autoencoder": DCBAAutoencoderWrapper,
    "supcon": DCBASupConWrapper,
}

_MODELS = {
    "ConfigAutoEncoder": ConfigAutoEncoder,
    "FeedforwardGraphConfigPredictor": FeedforwardGraphConfigPredictor,
}

_LOSSES: dict[str, type[nn.Module]] = {
    "mse": nn.MSELoss,
    "abcd_constraint": ABCDConstraintPenaltyLoss,
    "supcon": MultiPositiveSupConLoss,
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

    training_cfg = config["training"]
    wrapper_name = training_cfg["wrapper"]
    if wrapper_name not in _WRAPPERS:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")

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
        model_name = training_cfg["model_cls"]
        if model_name not in _MODELS:
            raise ValueError(f"Unknown model class '{model_name}'. Available: {list(_MODELS)}")
        encoder = _MODELS[model_name](**config["model"])
        loss_fn = _build_loss(training_cfg["loss"])
        wrapper: pl.LightningModule = DCBAAutoencoderWrapper(
            encoder,
            training_cfg["optimizer"]["args"],
            loss_fn=loss_fn,
        )
    elif wrapper_name == "supcon":
        losses_cfg = training_cfg["losses"]
        reg_loss = _build_loss(losses_cfg["reg"])
        supcon_loss = _build_loss(losses_cfg["repr"])
        if not isinstance(supcon_loss, MultiPositiveSupConLoss):
            raise ValueError(
                f"losses.repr must resolve to MultiPositiveSupConLoss, got {type(supcon_loss)}"
            )
        graph_encoder = FeedforwardGraphConfigPredictor(**config["model"])
        config_encoder = ConfigAutoEncoder(**config["config_model"])
        wrapper = DCBASupConWrapper(
            graph_encoder=graph_encoder,
            config_encoder=config_encoder,
            optimizer_config=training_cfg["optimizer"]["args"],
            reg_loss=reg_loss,
            supcon_loss=supcon_loss,
            lambda_supcon=training_cfg.get("lambda_supcon", 1.0),
        )
    else:
        raise ValueError(f"Unknown wrapper '{wrapper_name}'. Available: {list(_WRAPPERS)}")

    logger.log_hyperparams({key: value for key, value in config.items() if key != "hydra"})
    logger.watch(wrapper)
    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        accelerator=training_cfg["accelerator"],
        devices=training_cfg["devices"],
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
