"""Lightning callback factory."""

import json
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import wandb
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from lightning.pytorch.utilities.model_summary import summarize


def _base_state_key(callback: Callback, base: type[Callback]) -> str:
    """
    Return ``callback``'s state key as if it were an instance of ``base``.

    :param callback: The subclass instance whose key is being rewritten.
    :param base: The Lightning callback class whose key format should be mimicked.

    :returns: The state key string under the base class's name.
    """
    key = base.state_key.fget(callback)  # type: ignore[attr-defined]
    return key.replace(type(callback).__qualname__, base.__qualname__, 1)


class ConfigPatienceEarlyStopping(EarlyStopping):
    """``EarlyStopping`` that keeps the configured ``patience`` when resuming from a checkpoint."""

    #: Best monitored value inherited from the checkpoint; ``None`` in a run that started fresh.
    _inherited_best_score: float | None = None

    @property
    def state_key(self) -> str:
        """Mirror ``EarlyStopping``'s state key so checkpoints stay interoperable."""
        return _base_state_key(self, EarlyStopping)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore early-stopping counters, keeping the configured ``patience``."""
        configured_patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = configured_patience
        inherited = state_dict.get("best_score")
        self._inherited_best_score = None if inherited is None else float(inherited)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log the inherited record as a W&B artifact, once, for resumed runs only."""
        super().on_train_start(trainer, pl_module)
        if self._inherited_best_score is None:
            return
        run = getattr(trainer.logger, "experiment", None)
        if not isinstance(run, wandb.sdk.wandb_run.Run):
            return
        artifact = wandb.Artifact(name=f"resume_baseline_{run.id}", type="resume_baseline")
        with artifact.new_file("resume_baseline.json", mode="w") as f:
            json.dump(
                {
                    "monitor": self.monitor,
                    "best_score": self._inherited_best_score,
                    "patience_used": self.wait_count,
                    "patience": self.patience,
                },
                f,
            )
        run.log_artifact(artifact)


class RelocatableModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that keeps its top-k ranking when the run directory changes."""

    @property
    def state_key(self) -> str:
        """Mirror ``ModelCheckpoint``'s state key so checkpoints stay interoperable."""
        return _base_state_key(self, ModelCheckpoint)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the full top-k ranking, rebasing recorded paths onto the current ``dirpath``."""
        recorded_dirpath = state_dict.get("dirpath")
        if self.dirpath and recorded_dirpath and self.dirpath != recorded_dirpath:
            state_dict = {**state_dict, **self._rebased_paths(state_dict)}
        super().load_state_dict(state_dict)

    def _rebased_paths(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Build the state-dict entries naming checkpoint files, pointed at ``self.dirpath``."""

        def rebase(path: str) -> str:
            return str(Path(self.dirpath) / Path(path).name) if path else path

        rebased: dict[str, Any] = {"dirpath": self.dirpath}
        for key in ("best_model_path", "kth_best_model_path", "last_model_path"):
            if key in state_dict:
                rebased[key] = rebase(state_dict[key])
        if "best_k_models" in state_dict:
            rebased["best_k_models"] = {
                rebase(path): score for path, score in state_dict["best_k_models"].items()
            }
        return rebased


class WandbModelSummaryCallback(Callback):
    """Log model summary to W&B as an HTML panel and a text artifact."""

    def __init__(self, max_depth: int = -1) -> None:
        """:param max_depth: Depth passed to Lightning's ``summarize`` (``-1`` for all layers)."""
        self.max_depth = max_depth

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
                RelocatableModelCheckpoint(
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
