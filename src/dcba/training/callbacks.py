"""Lightning callback factory."""

from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint


def get_callbacks(config: dict) -> list[Callback]:
    """
    Build a list of Lightning callbacks from the training config.

    Supported callback names: ``model_checkpoint``, ``early_stopping``.

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
    return callbacks
