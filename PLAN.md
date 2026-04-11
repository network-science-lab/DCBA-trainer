# Enrich test output with config reconstruction table

## Steps

### 0. Accept scaler on the wrapper

`ConfigAutoencoderWrapper` currently has no access to the `ABCDConfigScaler`, which is needed to
inverse-transform normalised tensors back to human-readable values before logging.

Add an optional `scaler` parameter to `ConfigAutoencoderWrapper.__init__` in `src/dcba/wrapper.py`:

```python
def __init__(
    self,
    encoder: ConfigEncoder,
    optimizer_config: dict,
    scaler: ABCDConfigScaler | None = None,
) -> None:
```

Store it as `self._scaler`. Import `ABCDConfigScaler` from `dcba.dataset`.

### 1. Set the scaler after fit in the trainer

`datamodule.scaler` is populated inside `datamodule.setup()`, which Lightning calls during
`trainer.fit()` — so the scaler does not exist yet at wrapper construction time.

In `src/dcba/training/trainer.py`, between `trainer.fit()` and `trainer.test()`, assign the scaler
to the wrapper:

```python
trainer.fit(wrapper, datamodule=datamodule)
wrapper._scaler = datamodule.scaler
trainer.test(wrapper, datamodule=datamodule)
```

### 2. Accumulate per-sample rows during test_step

Override `on_test_epoch_start` in `ConfigAutoencoderWrapper` to reset a `self._test_rows`
accumulator (a plain `list`).

In `test_step`, after the forward pass, detach `x` and `x_hat`, move them to CPU, apply
`self._scaler.inverse_transform(...)` if a scaler is set, and append each sample in the batch as a
row to `self._test_rows`. Each row is a flat list of floats:

```
[orig_n, orig_t1, …, orig_nout, recon_n, recon_t1, …, recon_nout]
```

If `self._scaler` is `None`, log the raw (normalised) values with a note in the column names.

### 3. Log a wandb Table in on_test_epoch_end

Override `on_test_epoch_end` in `ConfigAutoencoderWrapper`. Build a `wandb.Table` whose columns are
`[f"orig_{k}" for k in ABCD_CONFIG_KEYS] + [f"recon_{k}" for k in ABCD_CONFIG_KEYS]` and whose data
is `self._test_rows`.

Log it via the Lightning logger's underlying wandb run:

```python
self.logger.experiment.log({"test/reconstructions": table})
```

Guard the wandb call: only execute it when `self.logger` has a real wandb experiment (i.e.
`hasattr(self.logger, "experiment")` and the experiment is not a `MagicMock`). Import
`ABCD_CONFIG_KEYS` from `dcba.dataset.transforms` and `wandb` at the top of `wrapper.py`.

Verify: after a training run, the wandb run page should show a `test/reconstructions` table with one
row per test sample and 18 float columns.
