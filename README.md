# Graph -> Config or DCBA

This repository contains research code addressing the problem of graph configuration retrieval, 
that is how to obtain parameters of a synthetic graph generator (e.g. ABCD) given the graph.

The problem is isnspired by evaluating what-if scenarios of complex networks which are vastly
modelled by networks. The idea, from the other hand, by CLIPs (Contrastive Language-Image
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

## Setting up the environment

(A) Clone the code into the persisting directory on your workstation, e.g., `XXX`

(B) Create container

1. Install docker with support for CUDA

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

## TODOs

- set up uv
- set up wandb project
- set up DVC
- set up pre-commit (and refresh it by migrating to ruff)
