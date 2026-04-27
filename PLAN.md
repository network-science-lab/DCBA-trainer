# Stage 1 SupCon Training

Implements `L = L_reg + λ · L_SupCon` for the graph-side encoder, where `L_reg` is MSE regression
from graph to ABCD config and `L_SupCon` is Multi-Positive Supervised Contrastive Loss with
soft-weighted negatives.

## Steps

### 0. Rename `GraphEncoder` → `FeedforwardGraphConfigPredictor` _(separate commit)_

The current model is a pure feedforward (no representation constraint) graph-to-config predictor —
rename to make that explicit and to contrast it with the upcoming SupCon-constrained variant. Use
`git mv`:

```
git mv src/dcba/models/graph_encoder.py src/dcba/models/feedforward_graph_config_predictor.py
```

Update everywhere:

- Class name inside the file
- `src/dcba/training/trainer.py` — `_MODELS` registry key and import
- `configs/gnn-encoder.yaml` — `model_cls: FeedforwardGraphConfigPredictor`
- Any other imports

Verify:
`uv run python -c "from dcba.models.feedforward_graph_config_predictor import FeedforwardGraphConfigPredictor"`.

### 1. Implement `MultiPositiveSupConLoss` + unit test _(separate commit)_

Add to `src/dcba/training/loss.py`. The loss operates purely on graph embeddings; the config tensor
is used only to soft-weight negatives.

**Signature:**

```python
class MultiPositiveSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, tau_dist: float = 1.0) -> None: ...

    def forward(
        self,
        embeddings: Tensor,   # (batch, embedding_dim) — h_G, L2-normalised inside
        labels: Tensor,        # (batch,) int — instance group index
        configs: Tensor,       # (batch, 9) — normalised θ, used for negative weighting
    ) -> Tensor: ...
```

**Algorithm:**

1. L2-normalise `embeddings`; compute cosine similarity matrix scaled by `temperature`
2. Build positive mask: `labels[i] == labels[j]` (excluding self)
3. Build negative soft-weight matrix: `w_ij = exp(−‖θ_i − θ_j‖₂ / tau_dist)` for
   `labels[i] != labels[j]`, else 0 — nearby configs are down-weighted as negatives
4. SupCon numerator: sum of `exp(sim_ij)` over positives
5. SupCon denominator: sum of `w_ij · exp(sim_ij)` over negatives (soft-weighted variant of the
   standard denominator)
6. Loss: mean over anchors of `−log(numerator / denominator)`

Export from `src/dcba/training/__init__.py`.

**Unit test** in `tests/training/test_loss.py`:

Cover at least:

- All samples share the same label → loss is well-defined (no valid negatives edge case)
- All samples have distinct labels → no positives, loss should raise or return a sentinel
- Soft-weighting: identical configs produce lower negative weight than distant configs
- Output is a non-negative scalar tensor with `requires_grad`

### 2. Add `DCBAStage1Wrapper` to `wrapper.py` _(single commit with step 3)_

```python
class DCBAStage1Wrapper(pl.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,           # FeedforwardGraphConfigPredictor instance
        optimizer_config: dict,
        lambda_supcon: float = 1.0,
        temperature: float = 0.07,
        tau_dist: float = 1.0,
        scaler: ABCDConfigScaler | None = None,
    ) -> None: ...
```

`_step` logic:

- Unpack `config`, `graph`, `target` from batch (same as `_unpack_batch`)
- Extract integer group labels from `batch.instance_id` (list of strings after PyG collation) —
  build a per-batch string-to-int mapping with `{uid: i for i, uid in enumerate(sorted(set(...)))}`
- Forward: `h_g, theta_hat = encoder((config, graph))`
- `l_reg = F.mse_loss(theta_hat, target)`
- `l_supcon = supcon_loss(h_g, labels, config)`
- `loss = l_reg + lambda_supcon * l_supcon`
- Log `{stage}_loss`, `{stage}_l_reg`, `{stage}_l_supcon` (all with `batch_size`)

Test step: same wandb Table as `DCBAAutoencoderWrapper` (original vs reconstructed θ).
`single_replica_per_instance` must be `False` — contrastive learning requires multiple graphs per
config to form positives within each batch.

### 3. Update `trainer.py` and add `configs/stage1-supcon.yaml` _(single commit with step 2)_

**`trainer.py`** additions:

- Register `"stage1_supcon": DCBAStage1Wrapper` in `_WRAPPERS`
- Register `"supcon": MultiPositiveSupConLoss` in `_LOSSES`
- Source `lambda_supcon`, `temperature`, `tau_dist` from `config["training"]` when building the
  wrapper

**`configs/stage1-supcon.yaml`** — mirror `gnn-encoder.yaml` with:

```yaml
data:
  report_path: "test/dataset_abcd/report.json" # interim dataset for initial runs
  batch_size: 16 # small enough for interim dataset; scale to ~256 (K≥32 configs × N graphs) for production

training:
  wrapper: stage1_supcon
  model_cls: FeedforwardGraphConfigPredictor
  lambda_supcon: 1.0
  temperature: 0.07
  tau_dist: 1.0
```
