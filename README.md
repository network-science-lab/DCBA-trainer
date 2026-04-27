# DCBA

This repository contains research code addressing the problem of **graph configuration retrieval**:
given a graph, recover the parameters of the synthetic generator (ABCD/mABCD) that produced it.

The problem is inspired by evaluating what-if scenarios affecting global topology of complex
networked systems. The idea draws from CLIP (Contrastive Language–Image Pre-Training).

## Idea

**Given a graph `G`, find the parameter vector `θ` of the ABCD/mABCD generator that could have
produced it.**

```
G -> Trained Model -> θ  (ABCD/mABCD parametrisation)
```

Use cases:

- Data augmentation
- Modelling macro-level interventions on a system
- Data compression

## Environment setup

**(A)** Clone the repository into a persistent directory on your workstation, e.g. `XXX`

**(B)** Create container

1. Install Docker with CUDA support.

2. Pull the image:

   ```bash
   docker pull ghcr.io/anty-filidor/mlaudio:gamma-uv
   ```

3. Run the container:

   ```bash
   docker run -itd \
       --gpus all \
       --name dcba \
       --shm-size=8gb \
       --cpus=16 \
       --memory=64g \
       --mount type=bind,source=/home/$USER/XXX,target=/workspace/XXX \
       ghcr.io/anty-filidor/mlaudio:gamma-uv
   ```

**(C)** Set up the Python project

1. Install `uv` dependencies:

   ```bash
   uv sync
   ```

2. Install `pre-commit`:

   ```bash
   uv run pre-commit install --config .pre-commit-config.yaml
   ```

3. To update `pre-commit` hooks, refer to
   [this template repository](https://github.com/anty-filidor/template-python).

## Run the code

Training:

```bash
uv run dcba-train --config-name base
```

Sweep (hyperparameter tuning):

```bash
export WANDB_DIR=.wandb
uv run wandb sweep ./configs/base-sweep.yaml
uv run wandb agent NAME --count X
```

## Model architecture

### Config encoder

- Stacked MLP autoencoder: `θ → h_θ → θ_hat`
- Kept as a full encoder–decoder pair — future direction: combine with a graph decoder to build a
  graph generator conditioned on `θ`

### Graph encoder

- GNN or Graphformer-based; ideally pretrained: `G → h_G`

## Training

### Dataset construction

Since the ABCD generator is stochastic, for each parameter vector `θ` we sample multiple graphs:

```
G_1, G_2, ..., G_n ~ p(G | θ)
```

This lets the model learn invariance to sampling noise and is what makes multi-positive contrastive
learning natural here (see below). Training pairs take the form `(G_i, θ)` for every sampled graph
(i.e. we don't group by `θ` in the training loop).

**Train/val/test split must be performed by `θ`**, not by individual graphs. Splitting by graph
leaks information — the model would have seen the same generator config during training and be
evaluated on its own memorised outputs rather than on genuinely unseen configurations.

### Pipeline overview

```text
      ┌──────────────────θ (group label)─────────────────┐
      │                                                  │
      v                                                  │
┌────────────┐              ┌──────────────┐             │
│  θ-Encoder │        G ──> │ Graph Encoder│ ──> h_G ────┤
│   θ → h_θ  │              └──────────────┘             │
└─────┬──────┘                              ┌────────────┴────────────┐
      │                                     │                         │
      v                              ┌──────┴──────┐       ┌──────────┴───────┐
┌────────────┐                       │ SupCon Loss │       │ Regression Head  │
│  θ-Decoder │                       │  h_G+ vs    │       │  h_G → θ_hat     │
│ h_θ → θ_hat│                       │{h_G+, h_G-} │       │  or (μ(G), σ²(G))│
└─────┬──────┘                       └─────────────┘       └─────────┬────────┘
      │                                                               │
  AE recon. loss                                               L_reg / L_NLL
```

### Losses

**Stage 1** — establish the basic signal

```
L = L_reg + λ · L_SupCon
```

- `L_reg` — MSE between predicted `θ_hat` and ground-truth `θ`; keeps the regression numerically
  grounded
- `L_SupCon` — [Multi-Positive Supervised Contrastive Loss](https://arxiv.org/abs/2004.11362) with
  cross-modal positives: each `h_G` is an anchor; its positives are the matching `h_θ` embedding
  (cross-modal, CLIP-style) **and** all other `h_G` embeddings from the same `θ` (same-modal
  multi-positive). Negatives are all `h_θ` and `h_G` embeddings from different `θ`. Both encoders
  are trained jointly; the shared `embedding_dim` is required.
  - negatives are **soft-weighted by parameter distance**: nearby configs are down-weighted rather
    than treated as hard negatives, which matters because `θ` is continuous
  - recommended batch: K≥32 configs × N graphs per config, e.g. 32×8=256 — contrastive losses
    require enough negatives to work well; K=16 is insufficient

**Stage 2** — add uncertainty awareness

```
L = L_NLL + λ · L_SupCon
```

Replace the regression head with a probabilistic one that predicts `(μ(G), σ²(G))` per parameter.
The [Gaussian NLL loss](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html)

```
L_NLL = (θ - μ)² / σ²  +  log σ²
```

forces the model to be honest about confidence: the `log σ²` term prevents it from inflating
variance to artificially suppress the prediction error. The predicted `σ²` captures **aleatoric
uncertainty** — the inherent ambiguity of the inverse problem.

**Stage 3** _(future)_

- keep dropout active at inference time and run K forward passes to estimate variance of `μ(G)`;
  this is a proxy for **epistemic uncertainty** — parameter regions the model has not seen enough of
- use these signals to drive adaptive generation of new training graphs in under-covered regions

## Links

- https://github.com/openai/CLIP — trained OpenAI CLIP
- https://github.com/mlfoundations/open_clip — open implementation of CLIP
