# VAE Config Encoder

Add a `ConfigVAE` alongside the existing `ConfigAutoEncoder` as an alternative theta encoder. The
VAE adds a reparameterised sampling step and a KL divergence loss, forcing the latent space into a
smooth, regularised geometry that provides a cleaner SupCon training signal for the graph encoder.
The plain AE is kept as a baseline for direct comparison.

## Steps

### 0. Add `ConfigVAE` model

Create `src/dcba/models/config_vae.py`. The class mirrors `ConfigAutoEncoder` in its public
interface — same constructor shape, same `encode` / `decode` / `forward` signatures — but the
encoder head splits into two linear projections (`mu` and `log_sigma`) sharing a common MLP trunk.

Key method signatures:

```python
def encode_distribution(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (mu, log_sigma) of shape (batch, embedding_dim) each."""

def encode(self, x: Tensor) -> Tensor:
    """Sample z via reparameterisation during training; return mu during eval."""

def decode(self, z: Tensor) -> Tensor: ...

def forward(self, batch: DCBAHeteroData) -> ForwardOutput: ...
```

`encode` must respect `self.training` — stochastic during training, deterministic at eval time — so
that SupCon inference is stable.

Export the class from `src/dcba/models/__init__.py` and register it in the `_MODELS` dict in
`src/dcba/training/trainer.py`.

### 1. Add `KLDivergenceLoss`

Create `src/dcba/training/loss/kl.py`. The module takes `(mu, log_sigma)` and returns the
closed-form KL divergence against N(0, 1), averaged over the batch:

```python
class KLDivergenceLoss(nn.Module):
    def forward(self, mu: Tensor, log_sigma: Tensor) -> Tensor:
        """Scalar KL divergence: mean over batch of -0.5 * sum(1 + 2·log_σ - μ² - σ²)."""
```

Export from `src/dcba/training/loss/__init__.py` and register as `"kl_divergence"` in the `_LOSSES`
dict in `trainer.py`.

### 2. Update `DCBASupConWrapper`

Add two optional constructor arguments:

```python
kl_loss: nn.Module | None = None
beta_kl: float = 0.01
```

In `_step`, after the existing loss computation, add an optional KL block:

```python
if self._kl_loss is not None:
    # encode_distribution is only available on ConfigVAE; plain ConfigAutoEncoder
    # never sets _kl_loss, so this branch is unreachable for the AE baseline.
    mu, log_sigma = self._config_encoder.encode_distribution(config)
    l_kl = self._kl_loss(mu, log_sigma)
    loss += self._beta_kl * l_kl
    self.log(f"{stage}_loss-kl", l_kl, batch_size=batch_size)
```

Include `kl_loss` and `beta_kl` in `save_hyperparameters(ignore=[...])` alongside the existing
ignored modules.

### 3. Wire up the trainer

In `trainer.py`, inside the `supcon` branch of `train()`, optionally build and pass the KL loss:

```python
kl_cfg = losses_cfg.get("kl")
kl_loss = _build_loss(kl_cfg) if kl_cfg is not None else None
wrapper = DCBASupConWrapper(
    ...
    kl_loss=kl_loss,
    beta_kl=kl_cfg.get("weight", 0.01) if kl_cfg is not None else 0.01,
)
```

No changes required when `kl` is absent from the config — the AE path is unaffected.

### 4. Add experiment config

Create `configs/gps-vae-supcon.yaml` by copying `configs/gps-supcon.yaml` and changing:

```yaml
models:
  theta:
    cls: ConfigVAE # was ConfigAutoEncoder
    input_dim: 9
    hidden_dims: [64, 32]
    embedding_dim: ${models.graph.embedding_dim}

training:
  losses:
    kl:
      name: kl_divergence
      weight: 0.01
  logger:
    name: mm-gps-vae-supcon
    tags: [gps, vae, community-features, abcd-constraint, bidirectional, batch64]
```

All other keys (`reg`, `repr`, graph encoder, datamodule) remain identical to the GPS baseline, so
results are directly comparable.
