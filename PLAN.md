# Stage 1 SupCon Training

Implements `L = L_reg + λ · L_SupCon` for joint training of `FeedforwardGraphConfigPredictor` and
`ConfigEncoder`. `L_reg` is MSE regression from graph embedding to ABCD config. `L_SupCon` is a
cross-modal Multi-Positive Supervised Contrastive Loss: graph embeddings are anchors, pulled towards
both the matching config embedding and other graph embeddings from the same generator config, and
pushed away from all embeddings (both modalities) from other configs.

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

Update tests if necessary.

### 1. Implement `MultiPositiveSupConLoss` + unit test _(separate commit)_

Add to `src/dcba/training/loss.py`.

**Contrastive setup:**

- Anchors: `h_G` embeddings (one per batch item, graph modality only)
- Positives for anchor `i`: `h_θ_i` (config embedding for the same `instance_id`) **+** `h_G_j` for
  all other `j` with the same `instance_id` — cross-modal and same-modal positives combined
- Negatives: all `h_θ_k` and `h_G_k` where `instance_id_k != instance_id_i`, soft-weighted by `θ`
  distance

Both encoders must share the same `embedding_dim` for the similarity computation to be valid.

**Signature:**

```python
class MultiPositiveSupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, tau_dist: float = 1.0) -> None: ...

    def forward(
        self,
        graph_embeddings: Tensor,   # (B, D) — h_G; these are the anchors
        config_embeddings: Tensor,  # (B, D) — h_θ; positives/negatives, not anchors
        labels: Tensor,             # (B,) int — per-sample group index; graph_embeddings[i]
                                    # and config_embeddings[i] share labels[i]
        configs: Tensor,            # (B, 9) — normalised θ, used for negative soft-weighting
    ) -> Tensor: ...
```

**Algorithm (canonical multi-positive SupCon with soft-weighted negatives):**

1. L2-normalise all embeddings; stack into a `(2B, D)` matrix `[h_G; h_θ]`, tile labels to `(2B,)` —
   rows `0..B-1` are graph anchors, rows `B..2B-1` are config embeddings
2. Compute `(B, 2B)` cosine similarity matrix between graph anchors and all embeddings, scaled by
   `temperature`; mask out the `(i, i)` self-similarity diagonal
3. Positive mask `(B, 2B)`: `tiled_labels[j] == labels[i]`, excluding self
4. Negative soft-weight matrix `(B, 2B)`: `w_ij = exp(-||θ_i - θ_j||_2 / tau_dist)` where
   `tiled_labels[j] != labels[i]`, else 0
5. For each anchor `i` and each positive `p` in `P(i)`:
   - numerator: `exp(sim(i, p) / τ)`
   - denominator: `Σ_{p' in P(i)} exp(sim(i, p') / τ)  +  Σ_{n in N(i)} w_in · exp(sim(i, n) / τ)`
   - term: `-log(numerator / denominator)`
6. Loss: mean over all `(i, p)` pairs

Export from `src/dcba/training/__init__.py`.

**Unit test** in `tests/training/test_loss.py`:

Cover at least:

- All samples share the same label → loss is well-defined (no valid negatives edge case)
- All samples have distinct labels → no positives, loss should raise or return a sentinel
- Soft-weighting: identical configs produce lower negative weight than distant configs
- Output is a non-negative scalar tensor with `requires_grad`

### 2. Add `DCBAStage1Wrapper` to `wrapper.py` _(single commit with step 3)_

Both encoders are trained jointly. The config encoder contributes `h_θ` as cross-modal positives and
negatives in `L_SupCon`; its reconstruction output is not used in this stage.

```python
class DCBAStage1Wrapper(pl.LightningModule):
    def __init__(
        self,
        graph_encoder: nn.Module,   # FeedforwardGraphConfigPredictor
        config_encoder: nn.Module,  # ConfigEncoder
        optimizer_config: dict,
        lambda_supcon: float = 1.0,
        temperature: float = 0.07,
        tau_dist: float = 1.0,
        scaler: ABCDConfigScaler | None = None,
    ) -> None: ...
```

`_step` logic:

- Unpack `config`, `graph`, `target` from batch (same as `_unpack_batch`)
- Extract integer group labels from `batch.instance_id` (list of strings after PyG collation):
  `{uid: idx for idx, uid in enumerate(sorted(set(batch.instance_id)))}`; convert to `(B,) int`
  tensor
- `h_g, theta_hat = graph_encoder((config, graph))`
- `h_theta, _ = config_encoder((config, graph))`
- `l_reg = F.mse_loss(theta_hat, target)`
- `l_supcon = supcon_loss(h_g, h_theta, labels, config)`
- `loss = l_reg + lambda_supcon * l_supcon`
- Log `{stage}_loss`, `{stage}_l_reg`, `{stage}_l_supcon` (all with `batch_size`)

Test step: same wandb Table as `DCBAAutoencoderWrapper` (original vs reconstructed θ from
`graph_encoder`). `single_replica_per_instance` must be `False`.

### 3. Update `trainer.py` and add `configs/stage1-supcon.yaml` _(single commit with step 2)_

**`trainer.py`** additions:

- Register `"stage1_supcon": DCBAStage1Wrapper` in `_WRAPPERS`
- Register `"supcon": MultiPositiveSupConLoss` in `_LOSSES`
- For `wrapper == "stage1_supcon"`, build both encoders from `config["model"]` (graph, using
  `FeedforwardGraphConfigPredictor`) and `config["config_model"]` (config, using `ConfigEncoder`),
  and pass `lambda_supcon`, `temperature`, `tau_dist` from `config["training"]`

**`configs/stage1-supcon.yaml`** — mirror `gnn-encoder.yaml` with:

```yaml
config_model:
  input_dim: 9
  hidden_dims: [64]
  embedding_dim: 32 # must match model.embedding_dim

model:
  input_dim: 1
  hidden_dims: [64, 128]
  embedding_dim: 32
  output_dim: 9

data:
  report_path: "test/dataset_abcd/report.json" # interim dataset for initial runs
  batch_size: 16 # small enough for interim dataset; scale to ~256 (K>=32 configs x N graphs) for production

training:
  wrapper: stage1_supcon
  model_cls: FeedforwardGraphConfigPredictor
  lambda_supcon: 1.0
  temperature: 0.07
  tau_dist: 1.0
```
