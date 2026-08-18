"""Unit tests for the Lightning callback factory."""

import torch
from lightning.pytorch.callbacks import EarlyStopping

from dcba.training.callbacks import ConfigPatienceEarlyStopping, get_callbacks


def _config(patience: int) -> dict:
    """Minimal training config with a single early-stopping callback."""
    return {
        "training": {
            "callbacks": [
                {
                    "name": "early_stopping",
                    "monitor": "val_loss",
                    "mode": "min",
                    "patience": patience,
                }
            ]
        }
    }


def test_early_stopping_is_built_from_config() -> None:
    """The factory wires monitor/mode/patience into the patience-preserving subclass."""
    (callback,) = get_callbacks(_config(patience=10))

    assert isinstance(callback, ConfigPatienceEarlyStopping)
    assert callback.monitor == "val_loss"
    assert callback.mode == "min"
    assert callback.patience == 10


def test_state_key_matches_vanilla_early_stopping() -> None:
    """The subclass must resolve to the same checkpoint slot as a plain ``EarlyStopping``."""
    (callback,) = get_callbacks(_config(patience=10))
    vanilla = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    assert callback.state_key == vanilla.state_key


def test_resume_keeps_configured_patience_and_restores_counters() -> None:
    """A checkpoint written with a shorter fuse must not override the configured patience."""
    original = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    original.wait_count = 2
    original.best_score = torch.tensor(0.0760)

    (resumed,) = get_callbacks(_config(patience=10))
    resumed.load_state_dict(original.state_dict())

    assert resumed.patience == 10
    assert resumed.wait_count == 2
    assert resumed.best_score == torch.tensor(0.0760)


def test_vanilla_early_stopping_would_have_regressed_patience() -> None:
    """Guard the assumption behind the subclass: Lightning restores ``patience`` from state."""
    original = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    vanilla = EarlyStopping(monitor="val_loss", mode="min", patience=10)
    vanilla.load_state_dict(original.state_dict())

    assert vanilla.patience == 3
