# Graph -> Config or DCBA

This repository contains research code addressing the problem of graph configuration retrieval, that
is how to obtain parameters of a synthetic graph generator (e.g. ABCD) given the graph.

The problem is isnspired by evaluating what-if scenarios of complex networks which are vastly
modelled by networks. The idea, from the other hand, stems from CLIPs (Contrastive Language-Image
Pre-Training).

## Idea

**Training**

```
Graph -> Embedding  .
                     \
                      => Contrastive learning -> Joint representation
                     /
Config -> Embedding .
```

**Inference**

```
Graph -> Trained Model -> Matching Config
```

**Use Cases**

- Data augmentation
- Modelling macro-level interventions on the system
- Data compression

## Environment setup

**(A)** Clone the code into the persisting directory on your workstation, e.g., `XXX`

**(B)** Create container

1. Install docker that supports CUDA

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

**(C)** Set up Python project

1. Install `uv` dependencies:

   ```bash
   uv sync
   ```

2. Install `pre-commit`:

   ```bash
   uv run pre-commit install --config .pre-commit-config.yaml
   ```

3. To update `pre-commit` visit [this](https://github.com/anty-filidor/template-python) repository.

## TODOs

- set up uv
- set up wandb project
- set up DVC
- set up pre-commit (and refresh it by migrating to ruff)
