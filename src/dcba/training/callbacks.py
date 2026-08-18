"""Lightning callback factory."""

import wandb
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from lightning.pytorch.utilities.model_summary import summarize


class ConfigPatienceEarlyStopping(EarlyStopping):
    """``EarlyStopping`` that keeps the configured ``patience`` when resuming from a checkpoint.
    """

    @property
    def state_key(self) -> str:
        """Mirror ``EarlyStopping``'s state key so checkpoints stay interoperable.
        """
        return f"EarlyStopping{ {'monitor': self.monitor, 'mode': self.mode}!r}"

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore early-stopping counters, keeping the configured ``patience``."""
        configured_patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = configured_patience


class WandbModelSummaryCallback(Callback):
    """Log model summary to W&B as an HTML panel and a text artifact."""

    def __init__(self, max_depth: int = -1) -> None:
        """:param max_depth: Depth passed to Lightning's ``summarize`` (``-1`` for all layers)."""
        self.max_depth = max_depth

    def on_train_start(self, trainer, pl_module) -> None:
        """Log the model summary once, at train start, when the logger is a W&B run."""
        run = getattr(trainer.logger, "experiment", None)
        if not isinstance(run, wandb.sdk.wandb_run.Run):
            return

        summary_str = str(summarize(pl_module, max_depth=self.max_depth))

        html = (
            "<pre style='font-family:monospace;white-space:pre;font-size:13px'>"
            f"{summary_str}"
            "</pre>"
        )
        run.log({"model/summary": wandb.Html(html)}, step=0)

        artifact = wandb.Artifact(name=f"model_summary_{run.id}", type="model_summary")
        with artifact.new_file("model_summary.txt", mode="w") as f:
            f.write(summary_str)
        run.log_artifact(artifact)


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
                ConfigPatienceEarlyStopping(
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
            callbacks.append(
                WandbModelSummaryCallback(max_depth=cb.get("max_depth", -1))
            )
    return callbacks
