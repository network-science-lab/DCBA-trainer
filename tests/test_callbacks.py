"""Unit tests for the Lightning callback factory."""

import json
from unittest.mock import MagicMock

import pytest
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dcba.training.callbacks import (
    ConfigPatienceEarlyStopping,
    RelocatableModelCheckpoint,
    get_callbacks,
)


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


def _checkpoint_config(dirpath: str) -> dict:
    """Minimal config with a single top-1 ``model_checkpoint`` callback writing to ``dirpath``."""
    return {
        "hydra": {"runtime": {"output_dir": dirpath}},
        "training": {
            "callbacks": [
                {
                    "name": "model_checkpoint",
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": True,
                }
            ]
        },
    }


def _checkpoint_state(dirpath: str, score: float, epoch: int) -> dict:
    """Build a ``ModelCheckpoint`` state dict as a run writing into ``dirpath`` would save it."""
    best = f"{dirpath}/checkpoints/model-epoch={epoch}.ckpt"
    return {
        "monitor": "val_loss",
        "best_model_score": torch.tensor(score),
        "best_model_path": best,
        "current_score": torch.tensor(score),
        "dirpath": f"{dirpath}/checkpoints",
        "best_k_models": {best: torch.tensor(score)},
        "kth_best_model_path": best,
        "kth_value": torch.tensor(score),
        "last_model_path": f"{dirpath}/checkpoints/last.ckpt",
    }


def test_model_checkpoint_state_key_matches_vanilla() -> None:
    """The subclass must resolve to the same checkpoint slot as a plain ``ModelCheckpoint``."""
    (callback,) = get_callbacks(_checkpoint_config("/runs/new"))
    vanilla = ModelCheckpoint(
        dirpath="/runs/new/checkpoints", monitor="val_loss", mode="min", save_top_k=1
    )

    assert isinstance(callback, RelocatableModelCheckpoint)
    assert callback.state_key == vanilla.state_key


def test_relocated_resume_keeps_ranking_and_rebases_paths() -> None:
    """Resuming into a new directory keeps the old scores, with paths pointing at the new one."""
    (resumed,) = get_callbacks(_checkpoint_config("/runs/new"))
    resumed.load_state_dict(_checkpoint_state("/runs/original", score=0.0760, epoch=10))

    assert resumed.best_model_score == torch.tensor(0.0760)
    assert resumed.kth_value == torch.tensor(0.0760)
    assert resumed.best_model_path == "/runs/new/checkpoints/model-epoch=10.ckpt"
    assert resumed.last_model_path == "/runs/new/checkpoints/last.ckpt"
    assert list(resumed.best_k_models) == ["/runs/new/checkpoints/model-epoch=10.ckpt"]


def test_same_directory_resume_is_untouched() -> None:
    """When the directory is unchanged the state is restored verbatim, as upstream would."""
    (resumed,) = get_callbacks(_checkpoint_config("/runs/same"))
    resumed.load_state_dict(_checkpoint_state("/runs/same", score=0.0894, epoch=22))

    assert resumed.best_model_score == torch.tensor(0.0894)
    assert resumed.best_model_path == "/runs/same/checkpoints/model-epoch=22.ckpt"


def test_vanilla_model_checkpoint_would_have_dropped_the_ranking() -> None:
    """Guard the assumption behind the subclass: upstream discards top-k on a dirpath change."""
    vanilla = ModelCheckpoint(
        dirpath="/runs/new/checkpoints", monitor="val_loss", mode="min", save_top_k=1
    )
    vanilla.load_state_dict(_checkpoint_state("/runs/original", score=0.0760, epoch=10))

    assert vanilla.best_model_score is None
    assert vanilla.best_k_models == {}


class _RecordingFile:
    """Stand-in for the file handle ``Artifact.new_file`` yields, minus the real filesystem I/O."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def __enter__(self) -> "_RecordingFile":
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def write(self, data: str) -> None:
        self._sink.append(data)


class _FakeArtifact:
    """Stand-in for ``wandb.Artifact`` that captures its single JSON file in memory."""

    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self._chunks: list[str] = []

    def new_file(self, filename: str, mode: str = "w") -> _RecordingFile:
        return _RecordingFile(self._chunks)

    def contents(self) -> dict:
        return json.loads("".join(self._chunks))


def _stub_trainer(monkeypatch: "pytest.MonkeyPatch", is_wandb: bool = True) -> MagicMock:
    """Build a trainer stub whose logger exposes a fake W&B run, and fake out ``wandb.Artifact``."""
    monkeypatch.setattr("dcba.training.callbacks.wandb.Artifact", _FakeArtifact)
    trainer = MagicMock()
    if is_wandb:
        trainer.logger.experiment = MagicMock(spec=wandb.sdk.wandb_run.Run)
        trainer.logger.experiment.id = "abc123"
    else:
        trainer.logger.experiment = object()
    return trainer


def test_resumed_run_logs_the_inherited_baseline_as_an_artifact(monkeypatch) -> None:
    """A resumed run must publish the record it is judged against, plus the patience burnt."""
    original = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    original.best_score = torch.tensor(0.0760)
    original.wait_count = 4

    (resumed,) = get_callbacks(_config(patience=10))
    resumed.load_state_dict(original.state_dict())

    trainer = _stub_trainer(monkeypatch)
    resumed.on_train_start(trainer, MagicMock())

    run = trainer.logger.experiment
    run.log_artifact.assert_called_once()
    (artifact,), _ = run.log_artifact.call_args
    assert artifact.name == "resume_baseline_abc123"
    assert artifact.type == "resume_baseline"
    assert artifact.contents() == {
        "monitor": "val_loss",
        "best_score": pytest.approx(0.0760, abs=1e-6),
        "patience_used": 4,
        "patience": 10,
    }


def test_fresh_run_logs_no_artifact(monkeypatch) -> None:
    """A run that started from scratch has no inherited record, so nothing extra is logged."""
    (callback,) = get_callbacks(_config(patience=10))

    trainer = _stub_trainer(monkeypatch)
    callback.on_train_start(trainer, MagicMock())

    trainer.logger.experiment.log_artifact.assert_not_called()


def test_non_wandb_logger_logs_no_artifact(monkeypatch) -> None:
    """A logger that isn't backed by a real W&B run (e.g. the offline dummy) is a silent no-op."""
    original = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    original.best_score = torch.tensor(0.0760)

    (resumed,) = get_callbacks(_config(patience=10))
    resumed.load_state_dict(original.state_dict())

    trainer = _stub_trainer(monkeypatch, is_wandb=False)
    resumed.on_train_start(trainer, MagicMock())  # must not raise
