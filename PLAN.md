# Config Autoencoder — Phase 1 Implementation

## Steps

### 0. Hardcode ABCD parameter bounds in `transforms.py`

The `ABCDConfigScaler` (see Step 1) needs per-feature `(lo, hi)` bounds to normalise each of the 9
config fields to `[0, 1]`. These bounds must be sourced from the actual ABCD model constraints
defined in the `dcba_data_set` package — **do not invent or guess them**.

Search the `dcba_data_set` repo for the authoritative bounds:

```bash
grep -r "t1\|t2\|xi\|c_min\|c_max\|d_min\|d_max\|nout\|n_min\|n_max" \
    /workspace/dev/DCBA-data-set/src --include="*.py" -n
```

Also read `/workspace/dev/DCBA-data-set/src/dcba_data_set/julia_ports/abcd.py` and the `ABCDConfig`
validators to confirm which fields have hard upper/lower limits.

Once found, define a module-level constant in `src/dcba/dataset/transforms.py`:

```python
#: Per-feature (lo, hi) bounds sourced from ABCDGraphGenerator.jl constraints.
ABCD_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "n":     (..., ...),
    "t1":    (..., ...),
    # etc.
}
```

---

### 1. Add `ABCDConfigScaler` to `transforms.py`

Add the scaler to the existing `src/dcba/dataset/transforms.py` — no new file needed.

The class takes optional `bounds` (defaults to `ABCD_PARAM_BOUNDS`) and exposes two methods:

- `transform(x: Tensor) -> Tensor` — normalise to `[0, 1]` per feature
- `inverse_transform(x: Tensor) -> Tensor` — recover original scale

The inverse must be exact (linear map), making it usable during inference to recover human-readable
configs from model output.

Export `ABCDConfigScaler` from `src/dcba/dataset/__init__.py`.

Update `DCBADataModule.setup()` to build an `ABCDConfigScaler` and compose it with
`ABCDConfigToTensor` into a single callable transform passed to each `ConfigDataset` split. Store
the scaler on the data module as `self.scaler` so the wrapper can retrieve it for inference.

---

### 2. Implement `ConfigEncoder`

Replace the stub in `src/dcba/models/config_encoder.py` with a parametrisable MLP autoencoder.

Constructor: `__init__(self, input_dim: int, hidden_dims: list[int], embedding_dim: int)`.

Architecture:

- Encoder: `input_dim → hidden_dims[0] → … → embedding_dim` with ReLU between layers
- Decoder: mirrors encoder in reverse, no activation on output

Public interface — keep `encode` and `decode` separate so the joint wrapper (Phase 2) can call them
independently:

- `encode(x: Tensor) -> Tensor`
- `decode(h: Tensor) -> Tensor`
- `forward(x: Tensor) -> tuple[Tensor, Tensor]` — returns `(h_q, x_hat)`

Suggested default: `hidden_dims=[64]`, `embedding_dim=32`. The `embedding_dim` is the key knob — it
must match whatever the graph encoder produces in Phase 2.

---

### 3. Add `ConfigAutoencoderWrapper` to `wrapper.py`

Phase 1 trains only the config autoencoder. Future phases will add a graph encoder and joint loss.
Use **separate `LightningModule` subclasses per training regime** — the `train()` function selects
the right one from config. This avoids mode flags and keeps each wrapper focused.

Rename/replace the existing stub with `ConfigAutoencoderWrapper(pl.LightningModule)`.

It should accept a `ConfigEncoder` and an `optimizer_config` dict, compute MSE reconstruction loss
in `training_step` / `validation_step` / `test_step`, and build an `AdamW` optimiser in
`configure_optimizers`. Log `{stage}_loss` at each step.

Future wrappers (`JointWrapper`, etc.) go in the same file and are selected via `training.wrapper`
in the Hydra config.

---

### 4. Wire up `train()` and complete `configs/base.yaml`

**`src/dcba/training/trainer.py`** — remove `raise NotImplementedError`. Build a `_WRAPPERS` dict
mapping string names to classes (e.g. `"config_autoencoder": ConfigAutoencoderWrapper`). Instantiate
`DCBADataModule`, `ConfigEncoder`, and the wrapper named in `config["training"]["wrapper"]`, then
call `trainer.fit()` and `trainer.test()`.

**`configs/base.yaml`** — fill in the currently empty `data` and `model` sections and add
`training.wrapper`:

```yaml
data:
  report_path: ??? # must be overridden at runtime
  val_ratio: 0.1
  test_ratio: 0.1
  batch_size: 32
  num_workers: 0

model:
  input_dim: 9
  hidden_dims: [64]
  embedding_dim: 32

training:
  wrapper: config_autoencoder
  # ... rest unchanged
```

Run:

```bash
uv run dcba-train data.report_path=/path/to/report.json
```
