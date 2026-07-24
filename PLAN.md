# Untangle scalers/penalty loss, drop KL/VAE

Eight commits, each meant to land independently with tests green. KL/VAE removal goes first (YAGNI —
simplifies the wrapper files before the scaler-agnostic penalty loss work touches them). Each "stop
using" commit is separated from its "delete the now-dead code" commit so review stays small.

## Commits

### 1. Stop wiring KL/VAE into the wrappers and trainer

Remove the _usage_ of the KL/VAE path without deleting the underlying classes yet (kept for commit
2, so nothing else needs to change in this one):

- `src/dcba/wrappers/supcon.py`: drop `KLRegularisedMixin` from `DCBASupConWrapper`'s bases; remove
  the `kl_loss`/`beta_kl`/`kl_warmup_epochs` constructor params; simplify `_step`/`encode()` so
  `z_theta` always comes from `self._config_encoder.encode(config)` — delete the
  `if self._kl_loss is not None: mu, log_sigma = ...` branch and the `_kl_term` call.
- `src/dcba/wrappers/autoencoder.py`: same for `DCBAAutoencoderWrapper` — drop the mixin, the three
  constructor params, and the `if self._kl_loss is not None` branch in `_step`.
- `src/dcba/training/trainer.py`: remove the `kl_cfg`/`kl_loss` wiring block from both
  `build_supcon_wrapper` and the `config_autoencoder` branch of `train()` (the `ConfigVAE`
  isinstance checks and the `kl_loss=...`/`beta_kl=...`/`kl_warmup_epochs=...` kwargs passed to the
  wrappers).

Verify: `uv run pytest tests/ -q` (the `TestKLDivergenceLoss`/`TestKLRegularisedMixin` tests in
`tests/test_loss.py` still pass — they exercise the loss/mixin classes directly, not the wrappers).

### 2. Delete the now-dead KL/VAE code

- Delete `src/dcba/models/config_vae.py` and `src/dcba/training/loss/kl.py`.
- `src/dcba/models/__init__.py`: remove the `ConfigVAE` import and `__all__` entry.
- `src/dcba/training/loss/__init__.py`: remove the `KLDivergenceLoss` import and `__all__` entry.
- `src/dcba/training/trainer.py`: remove the `ConfigVAE`/`KLDivergenceLoss` imports, the
  `"ConfigVAE"` entry in `_MODELS`, and the `"kl_divergence"` entry in `_LOSSES`.
- `src/dcba/wrappers/base.py`: delete the `KLRegularisedMixin` class entirely.
- `src/dcba/models/gin_encoder.py`: drop the docstring's dangling reference to `ConfigVAE`.
- `tests/test_loss.py`: remove `TestKLDivergenceLoss`, `TestKLRegularisedMixin`, `_KLMixinStub`, and
  the now-unused imports.

Verify: `uv run pytest -q` and
`grep -rn "ConfigVAE\|KLDivergenceLoss\|KLRegularisedMixin\|kl_loss\|beta_kl\|kl_warmup" src tests`
returns nothing.

### 3. Move scaler code out of `transforms.py` into `scalers.py` (pure move, no behaviour change)

`src/dcba/dataset/transforms.py` mixes node-feature transforms (`CommunityToSize`,
`ConstantNodeFeatures`) with theta-vector scaling/encoding. Move everything except those two classes
into a new `src/dcba/dataset/scalers.py`, verbatim: `ABCD_CONFIG_KEYS`, `ABCD_INT_FEATURE_INDICES`,
`ABCDConfigSchema`, `abcd_param_bounds`, `abcd_nmax_bounds`, `ABCDConfigScaler`,
`ABCDNMaxConfigScaler`, `ABCDLogConfigScaler`, `ABCDRelativeConfigScaler`, `ABCDConfigToTensor`.
No dedup yet — that's commit 4, kept separate so this commit is a reviewable pure
file move.

Update every import site currently doing `from dcba.dataset.transforms import ...` for one of the
moved symbols, to `from dcba.dataset.scalers import ...`:

- `src/dcba/dataset/dcba_dataset.py:16` — split into two imports (`CommunityToSize` stays from
  `transforms`, `ABCDConfigScaler`/`ABCDConfigToTensor` move to `scalers`)
- `src/dcba/training/trainer.py:21`
- `src/dcba/wrappers/autoencoder.py:13`
- `src/dcba/wrappers/supcon.py:16`
- `tests/test_scalers.py:7`
- `tests/test_loss.py` (7 inline imports at lines 924, 978, 995, 1006, 1017, 1034, 1056)
- `scripts/embedding_stability.py:57`
- `scripts/check_config_scaling.py:35`

Update `src/dcba/dataset/__init__.py` to import node transforms from `.transforms` and everything
else from `.scalers`, keeping the same `__all__` (so `from dcba.dataset import X` call sites are
unaffected).

Verify: `uv run pytest tests/test_scalers.py tests/test_dataset_loading.py -q`.

### 4. Introduce `ABCDBaseConfigScaler` and dedupe the 4 scalers onto it

In `src/dcba/dataset/scalers.py`, add a concrete base class factoring out what's currently
copy-pasted 4 times:

```python
class ABCDBaseConfigScaler:
    """Shared scaffolding for [0, 1] ABCD config scalers."""

    def transform(self, x: Tensor) -> Tensor: ...
    def denormalise(self, x: Tensor) -> Tensor: ...

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Concrete: calls denormalise then rounds ABCD_INT_FEATURE_INDICES. Same in all 4 today."""

    def __call__(self, x: Tensor) -> Tensor:
        """Concrete: return self.transform(x)."""

    def _scale_c_min(self, c_min_raw: Tensor, c_max_raw: Tensor) -> Tensor: ...
    def _unscale_c_min(self, c_min_norm: Tensor, c_max_raw: Tensor) -> Tensor: ...
```

`transform`/`denormalise` stay abstract (each subclass's per-feature maths genuinely differs). Make
`ABCDConfigScaler`, `ABCDNMaxConfigScaler`, `ABCDLogConfigScaler`, `ABCDRelativeConfigScaler`
inherit from it, deleting their own `inverse_transform`/`__call__` and the repeated
`c_min = c_min_raw / c_max_raw` inline maths in favour of the shared helpers. Behaviour must be
identical — this is a refactor, not a maths change.

Verify: `uv run pytest tests/test_scalers.py -q` — every existing test should pass unchanged since
outputs are identical.

### 5. Add `ABCDIdentityConfigScaler`

A 5th scaler in `scalers.py`, inheriting `ABCDBaseConfigScaler`: `transform`/`denormalise` are the
identity (`return x`); `inverse_transform`/`__call__` are inherited unchanged, so integer features
still get rounded. This gives the "no scaling" ablation arm a real scaler object instead of `None` —
needed because commit 7 makes a scaler mandatory for `ABCDConstraintPenaltyLoss`.

- Register it in `src/dcba/training/trainer.py`'s `_SCALERS` dict as `"ABCDIdentityConfigScaler"`.
- Add it to `src/dcba/dataset/__init__.py`'s imports/`__all__` alongside `ABCDBaseConfigScaler`.
- `configs/gps-ae-supcon-no-transform.yaml` currently omits `data.scaler` entirely (so
  `_build_scaler(None) -> None`) — set `data.scaler: {name: ABCDIdentityConfigScaler}` explicitly.

Verify:
`uv run python -c "from dcba.dataset import ABCDIdentityConfigScaler; import torch; s = ABCDIdentityConfigScaler(); print(s.transform(torch.ones(9)))"`.

### 6. Dedupe scaler instantiation in `trainer.py`

`_attach_ordering_scaler` (lines 168-186) currently builds a _second_ scaler instance from config
purely for the loss, independent of the one `train()` already builds for the datamodule (line 247).
Change it to take the already-built scaler directly:

```python
def _attach_ordering_scaler(loss: nn.Module, scaler: ABCDConfigScaler | None) -> None:
    """Attach `scaler` to `loss` when it's an ABCDConstraintPenaltyLoss."""
```

Thread the single `scaler` built in `train()` through to both call sites (`build_supcon_wrapper`'s
`reg_loss` and the `config_autoencoder` branch's `loss_fn`) instead of each rebuilding it from
`config["data"]["scaler"]`. `build_supcon_wrapper` is also called externally
(`scripts/embedding_stability.py`) without a pre-existing scaler — give it an optional
`scaler: ABCDConfigScaler | None = None` parameter that builds one internally via `_build_scaler`
when not supplied, so that call site keeps working unchanged. This commit is behaviour-preserving
(same scaler config, just one instance instead of two) — no test changes expected.

Verify: `uv run pytest tests/ -q`.

### 7. `ABCDConstraintPenaltyLoss` raw-only cutover

`src/dcba/training/loss/penalty.py`'s `ordering_penalties` argument (`"scaled"`/`"raw"`/`"none"`)
hardcodes per-scaler caveats in its docstring, and only `"raw"` is documented as correct under any
scaler. Remove the mode entirely:

- Delete `_ORDERING_MODES`, the `ordering_penalties` constructor argument, `self.ordering_mode`, and
  the `"scaled"`/`"none"` branches in `_ordering_penalty` — keep only the current `"raw"` logic
  (denormalise `x_hat` and `target`, divide by target's raw `n`).
- `set_scaler` becomes mandatory before the first `forward()`: raise the existing `RuntimeError`
  whenever `self._scaler is None`, unconditionally. This is safe now that every config attaches a
  real scaler (commit 5's `ABCDIdentityConfigScaler` covers the "no scaling" arm).
- Delete the local `_DenormalisingScaler` `Protocol` (confirmed referenced nowhere outside this
  file) — type `_scaler`/`set_scaler` directly against `dcba.dataset.scalers.ABCDBaseConfigScaler`
  instead.
- Update the class docstring to drop the per-scaler-subclass caveats — raw mode is always correct
  now, including for the identity scaler.
- Remove `ordering_penalties` from every config's `training.losses.reg.args` — currently set in all
  six `configs/gps-ae-supcon-*.yaml` files (`base-scaler`, `log-per-instance-scaler`,
  `no-transform`, `relative-scaler`, `n10k-scaler`, `log-scaler`).

Verify: `uv run pytest tests/test_loss.py -k ABCDConstraintPenalty -q`.

### 8. Full verification pass

```bash
uv run pytest -q
uv run dcba-train --config-name gps-ae-supcon-relative-scaler training.max_epochs=1
uv run dcba-train --config-name gps-ae-supcon-no-transform training.max_epochs=1
```

The second training run exercises `ABCDIdentityConfigScaler` end to end through the now-mandatory
penalty-loss scaler path. Confirm `REVIEW_REMARKS.md`'s excluded scaler/penalty-loss remarks are now
moot rather than re-litigating them — this refactor supersedes that discussion.
