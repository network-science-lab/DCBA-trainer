# Repository architecture: Lightning + wandb scaffold (phased)

## Overview

Build the repo in phases so each one is a runnable experiment before the next begins:

- **Phase 1 (this plan):** config autoencoder — `q → h_q → q_hat`, trained on configs alone
- **Phase 2 (future):** add graph encoder and align `h_q ~ h_g`
- **Phase 3 (future):** contrastive loss (InfoNCE), then inference `g → h_g → q_hat`

## Steps

### 1. Add dependencies

```bash
uv add lightning wandb hydra-core omegaconf
```

Verify:

```bash
uv run python -c "import lightning, wandb, hydra; print('ok')"
```

### 2. Build module skeleton

Create the directory structure. Anything not needed for Phase 1 gets an empty stub — it exists so
imports resolve, but contains only `raise NotImplementedError`.

**Directories** (each needs an `__init__.py`):

```
src/dcba/utils/
src/dcba/models/
src/dcba/training/
```

**Move `paths.py` into `utils/`:**

```bash
git mv src/dcba/paths.py src/dcba/utils/paths.py
```

Update the one import this breaks, in `tests/test_dataset_loading.py`:

```python
# before
from dcba.paths import TEST_ABCD_REPORT, TEST_MABCD_REPORT
# after
from dcba.utils.paths import TEST_ABCD_REPORT, TEST_MABCD_REPORT
```

**Stub files** (Phase 2+ — `raise NotImplementedError` bodies only):

| File                               | What it will become        |
| ---------------------------------- | -------------------------- |
| `src/dcba/models/graph_encoder.py` | `GraphEncoder(nn.Module)`  |
| `src/dcba/training/loss.py`        | `IdeaALoss`, `InfoNCELoss` |

Verify tests still pass:

```bash
uv run pytest tests/test_dataset_loading.py
```

### 3. Implement training infrastructure

The files below are fully implemented. The model files (`config_encoder.py`, `wrapper.py`) remain
stubs — they will be filled in separately once the scaffold is in place.

---

**`src/dcba/utils/config.py`** — Hydra config helpers:

```python
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

def load_config(cfg: DictConfig) -> dict:
    config = OmegaConf.to_container(cfg, resolve=True)
    config["hydra"] = OmegaConf.to_container(HydraConfig.get(), resolve=True)
    return config
```

---

**`src/dcba/dataset.py`** — `DCBAConfigDataset(torch.utils.data.Dataset)` stub.

Note for implementation: the ABCD config YAML contains these numerical keys (verified against
`abcd-interim`): `n, t1, t2, xi, c_min, c_max, d_min, d_max, nout` — 9 features total. The dataset
should load a report with `load_report()`, extract a float tensor per `ConfigRecord`, and return
`(tensor, tensor)` (input = target for the autoencoder). Split must be on `instance_id`, not on
individual replicas, to prevent data leakage.

---

**`src/dcba/datamodule.py`** — `DCBADataModule(pl.LightningDataModule)` stub.

Note for implementation: `__init__` accepts `report_path`, `val_ratio`, `test_ratio`, `batch_size`,
`num_workers`; `setup(stage)` builds and splits `DCBAConfigDataset`; standard `train_dataloader`,
`val_dataloader`, `test_dataloader`.

---

**`src/dcba/models/config_encoder.py`** — stub only (`raise NotImplementedError`).

**`src/dcba/wrapper.py`** — stub only (`raise NotImplementedError`).

---

**`src/dcba/training/callbacks.py`** — `get_callbacks(config: dict) -> list`:

Mirror the pattern from `infmax-trainer`: match on `callback["name"]`, support `model_checkpoint`
and `early_stopping` for now.

---

**`src/dcba/training/loggers.py`** — `get_logger(config: dict)`:

```python
from lightning.pytorch import loggers

class _DummyLogger:
    def __getattr__(self, name):
        return lambda *a, **kw: None

def get_logger(config: dict) -> loggers.WandbLogger | _DummyLogger:
    try:
        return loggers.WandbLogger(
            project=config["training"]["logger"]["project"],
            name=config["training"]["logger"].get("name"),
            tags=config["training"]["logger"].get("tags", []),
            save_dir=config["hydra"]["run"]["dir"],
        )
    except Exception as exc:
        logging.warning("WandbLogger not initialised — using dummy. Reason: %s", exc)
        return _DummyLogger()
```

---

**`src/dcba/training/trainer.py`** — `train(config: dict)` stub.

The function signature and `pl.Trainer` wiring should be in place; the model instantiation lines are
left as `raise NotImplementedError` until `wrapper.py` is implemented:

```python
def train(config: dict) -> None:
    raise NotImplementedError  # fill in once wrapper.py is implemented

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["training"]["accelerator"],
        devices=config["training"]["devices"],
        log_every_n_steps=1,
        callbacks=get_callbacks(config),
        logger=get_logger(config),
    )
    # trainer.fit(wrapper, datamodule=datamodule)
    # trainer.test(wrapper, datamodule=datamodule)
```

### 4. Wire the Hydra entrypoint

**`configs/hydra.yaml`:**

```yaml
hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**`configs/base.yaml`** — only what is known now; the rest is filled in once the model and dataset
are implemented:

```yaml
base:
  random_seed: 42

data: {} # TBD

model: {} # TBD

training:
  accelerator: gpu
  devices: [0]
  max_epochs: 100
  optimizer:
    name: AdamW
    args:
      lr: 0.001
      weight_decay: 0.00001
  logger:
    project: dcba
    name: config-autoencoder
    tags: [phase-1]
  callbacks:
    - name: model_checkpoint
      monitor: val_loss
      mode: min
      save_top_k: 1
      save_last: true
      verbose: true
    - name: early_stopping
      monitor: val_loss
      mode: min
      patience: 20
```

**`src/dcba/train.py`:**

```python
"""Training entrypoint — invoke via `uv run dcba-train`."""

from pathlib import Path
import hydra
from omegaconf import DictConfig

_CONFIGS_PATH = Path(__file__).parent.parent.parent / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIGS_PATH), config_name="base")
def main(cfg: DictConfig) -> None:
    from dcba.training.trainer import train
    from dcba.utils.config import load_config
    config = load_config(cfg)
    train(config)


if __name__ == "__main__":
    main()
```

**`src/dcba/__init__.py`** — replace placeholder body:

```python
"""DCBA: graph configuration retrieval via contrastive learning."""
from dcba.train import main
__all__ = ["main"]
```

**`pyproject.toml`** — update `[project.scripts]`:

```toml
[project.scripts]
dcba-train = "dcba.train:main"
```

Verify wandb connectivity by logging a dummy experiment before running the full stack:

```bash
uv run python - <<'EOF'
import wandb

wandb.login()
run = wandb.init(project="dcba", name="connectivity-test", tags=["sanity"])
for step in range(10):
    run.log({"train_loss": 1.0 / (step + 1), "val_loss": 1.2 / (step + 1)})
run.finish()
print("wandb ok")
EOF
```

This will prompt for an API key if not already in the environment. The run will appear in the wandb
UI under the `dcba` project with a small loss curve — confirm it renders correctly, then delete it.

Then verify the Hydra entrypoint resolves (dry run, CPU, 1 epoch — `trainer.fit` is still stubbed so
this only checks imports and config loading):

```bash
uv run dcba-train training.accelerator=cpu training.devices=1 training.max_epochs=1
```
