"""
Connectivity smoke test for Weights & Biases logging.

run it as: `uv run pytest -m wandb -v`
"""

import pytest
import wandb

from dcba.training.loggers import DummyLogger, get_logger

wandb_only = pytest.mark.wandb

_CONFIG = {
    "training": {
        "logger": {
            "project": "dcba",
            "name": "connectivity-test",
            "tags": ["sanity"],
        },
    },
    "hydra": {
        "run": {"dir": ".wandb"},
    },
}


@wandb_only
class TestWandbConnectivity:
    """Verify that get_logger returns a live WandbLogger and can log metrics."""

    def test_login(self) -> None:
        """wandb.login() succeeds without prompting (API key must be pre-configured)."""
        result = wandb.login(anonymous="never", timeout=30)
        assert result, "wandb login failed — check WANDB_API_KEY or ~/.netrc"

    def test_get_logger_returns_wandb_logger(self) -> None:
        """get_logger returns a WandbLogger, not the DummyLogger fallback."""
        from lightning.pytorch.loggers import WandbLogger

        logger = get_logger(_CONFIG)
        assert not isinstance(logger, DummyLogger), (
            "get_logger fell back to DummyLogger — check WANDB_API_KEY or ~/.netrc"
        )
        assert isinstance(logger, WandbLogger)
        logger.experiment.finish()

    def test_log_and_finish(self) -> None:
        """A WandbLogger can log metrics and finish cleanly."""
        from lightning.pytorch.loggers import WandbLogger

        logger = get_logger(_CONFIG)
        assert isinstance(logger, WandbLogger)

        for step in range(10):
            logger.log_metrics(
                {"train_loss": 1.0 / (step + 1), "val_loss": 1.2 / (step + 1)},
                step=step,
            )

        run = logger.experiment
        run.finish()
        assert run.id is not None, "Run did not finish with a valid ID"
