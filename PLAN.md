# ABCD Constraint Penalty Loss & Scaler Dtype Fix

## Steps (each step each commit)

### 1. Fix `ABCDConfigScaler.inverse_transform` dtypes

In `src/dcba/dataset/transforms.py`, add a module-level constant listing the indices of
integer-valued ABCD features:

```python
#: Indices within ABCD_CONFIG_KEYS that correspond to integer-valued parameters.
ABCD_INT_FEATURE_INDICES: list[int] = [0, 4, 5, 6, 7, 8]  # n, c_min, c_max, d_min, d_max, nout
```

Then update `ABCDConfigScaler.inverse_transform` to round those features to the nearest integer
after the linear inverse map. The tensor dtype stays `float32`; rounding ensures that integer fields
(e.g. `n`) surface as whole numbers in downstream `.tolist()` calls and wandb tables.

Export `ABCD_INT_FEATURE_INDICES` from `src/dcba/dataset/__init__.py`.

Verify: instantiate `ABCDConfigScaler`, call `.inverse_transform` on a normalised tensor, and
confirm that positions 0, 4, 5, 6, 7, 8 have no fractional part.

### 2. Implement `ABCDConstraintPenaltyLoss`

In `src/dcba/training/loss.py`, implement the new loss class. All arithmetic is done in
**normalised** space (values expected in `[0, 1]`), which is what the model produces and trains on.

The loss is: `MSE(x_hat, target) + λ * Σ penalties`, where each penalty term is `relu(violation)²`.

Two categories of penalty:

**Per-feature range** — applied to all 9 features of `x_hat`:

- Below-range: `relu(-x_hat[:, i])`
- Above-range: `relu(x_hat[:, i] - 1)`

**Cross-parameter ordering** — using feature indices from `ABCD_CONFIG_KEYS`: | Constraint |
Violation expression | |---|---| | `c_min ≤ c_max` | `relu(x_hat[:, 4] - x_hat[:, 5])` | |
`d_min ≤ d_max` | `relu(x_hat[:, 6] - x_hat[:, 7])` | | `c_max ≤ n` |
`relu(x_hat[:, 5] - x_hat[:, 0])` | | `nout ≤ n` | `relu(x_hat[:, 8] - x_hat[:, 0])` |

Interface:

```python
class ABCDConstraintPenaltyLoss(nn.Module):
    """
    MSE reconstruction loss augmented with squared-hinge penalties for ABCD config constraints.

    :param lambda_penalty: Weight applied to the sum of constraint penalty terms.
    """

    def __init__(self, lambda_penalty: float = 1.0) -> None: ...

    def forward(self, x_hat: Tensor, target: Tensor) -> Tensor:
        """
        Compute MSE + λ · Σ relu(violation)².

        :param x_hat: Reconstructed normalised config tensor of shape ``(batch, 9)``.
        :param target: Ground-truth normalised config tensor of shape ``(batch, 9)``.

        :returns: Scalar loss tensor.
        """
```

Export the class from `src/dcba/training/__init__.py`.

### 3. Accept `loss_fn` in `ConfigAutoencoderWrapper`

In `src/dcba/wrapper.py`, add an optional `loss_fn: nn.Module | None = None` parameter to
`ConfigAutoencoderWrapper.__init__`. When `None`, fall back to `F.mse_loss` (current behaviour).
Store it as `self._loss_fn`.

Update `_step` to call `self._loss_fn(x_hat, target)` when a loss function is set, otherwise keep
the existing `F.mse_loss` call.

### 4. Wire loss selection through `trainer.py` and `configs/base.yaml`

Add a `_LOSSES` registry in `src/dcba/training/trainer.py` mapping string names to loss
constructors:

```python
_LOSSES: dict[str, type[nn.Module]] = {
    "mse": None,  # sentinel — wrapper uses F.mse_loss
    "abcd_constraint": ABCDConstraintPenaltyLoss,
}
```

Read `config["training"]["loss"]` (name + args) in `train()`, instantiate the loss if it is not
`mse`, and pass it to the wrapper constructor.

Add the `loss` key to `configs/base.yaml` with `mse` as the default, plus a commented-out example
for `abcd_constraint`:

```yaml
training:
  loss:
    name: mse
    args: {}
  # To use the constraint-penalty loss:
  # loss:
  #   name: abcd_constraint
  #   args:
  #     lambda_penalty: 1.0
```

Verify the full pipeline end-to-end:

```bash
uv run dcba-train --config-name base
uv run dcba-train --config-name base 'training.loss.name=abcd_constraint' \
    'training.loss.args.lambda_penalty=1.0'
```

Both runs should complete without error and log `train_loss` / `val_loss` to wandb.
