"""Lightning callback factory."""

from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
    ModelSummary,
)


def get_callbacks(config: dict) -> list[Callback]:
    """
    Build a list of Lightning callbacks from the training config.

    Supported callback names: ``model_checkpoint``, ``early_stopping``, ``model_summary``.

    :param config: Full resolved training config dict (as returned by
        :func:`~dcba.utils.config.load_config`).

    :returns: A list of instantiated :class:`~lightning.pytorch.callbacks.Callback` objects.
    """
    callbacks: list[Callback] = []
    for cb in config.get("training", {}).get("callbacks", []):
        name = cb["name"]
        if name == "model_checkpoint":
            callbacks.append(
                ModelCheckpoint(
                    dirpath=f"{config['hydra']['runtime']['output_dir']}/checkpoints/",
                    filename="model-{epoch}",
                    monitor=cb.get("monitor"),
                    mode=cb.get("mode", "min"),
                    save_top_k=cb.get("save_top_k", 1),
                    save_last=cb.get("save_last", False),
                    verbose=cb.get("verbose", False),
                )
            )
        elif name == "early_stopping":
            callbacks.append(
                EarlyStopping(
                    monitor=cb.get("monitor"),
                    mode=cb.get("mode", "min"),
                    patience=cb.get("patience", 10),
                )
            )
        elif name == "gradient_accumulation_scheduler":
            gradient_scheduling = cb["gradient_scheduling"]
            scheduling = (
                gradient_scheduling
                if isinstance(gradient_scheduling, dict)
                else {0: gradient_scheduling}
            )
            callbacks.append(GradientAccumulationScheduler(scheduling=scheduling))
        elif name == "model_summary":
            callbacks.append(ModelSummary(max_depth=cb.get("max_depth", -1)))
    return callbacks
