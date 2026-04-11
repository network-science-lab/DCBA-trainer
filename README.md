# DCBA

This repository contains research code addressing the problem of graph configuration retrieval, that
is how to obtain parameters of a synthetic graph generator (e.g. ABCD) given the graph.

The problem is isnspired by evaluating what-if scenarios of complex networks which are vastly
modelled by networks. The idea, from the other hand, stems from CLIPs (Contrastive Language-Image
Pre-Training).

## Idea

**Given a graph `g` find a set of parameters `q` according to which it can be generated with ABCD.**

```
Graph -> Trained Model -> Matching Config
```

Use Cases:

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

# Doodles

## Data

- pairs config, graph: `(q, g)`
- the configurational space should not be too large
- graphs also shouldn't be too big, but as for me no smaller than 1000 nodes
- use diverse configuration

## Architecture

### Config Encoder

- use a simple autoencoder
- can be a stacked MLP (?)

```
q - config
h_q - config's emgedding

q -> h_q -> q_hat
```

### Graph Encoder

- take a graphformer's instance (?)
- should be a pretrained model as these guys are big

```
g - graph
h_g - graph's embedding

g -> h_g
```

## Training

- aimed to align embeddings of pairs (q, g) thus make q_hat sensible given the graph
- after training we expect that: h_q ~ h_g

### Idea A

- easier than CLIP-based approaches, can be too shallow, but does not require so much data
- joined training of both encoders
- minimise the loss:

```
L = \lambda_1 ||h_q - h_g||^2 + \lambda_2 ||q_hat - q||^2
```

- intuition is that the loss is a weighted error of discrepancy between corresponding config-based
  and graph based embeddings plus weighted error of retrieving the configuration by the autoencoder

### Idea B

- derived from CLIPs, requires large training batches (min 256)
- contrastive learning par excellence
- minimise the loss:

```
L = InfoNCE(q, g)

1. compute cross-modal cosine similarity in a batch
2. store similarities in a square matrix, with row-indices matching h_q and column-indices matching h_g and ordered by graph-config equivalence
3. for each row try to make diagonal entry bigger than the sum of latter entries
4. dito for columns
```

## TODOs

invoke training: `uv run dcba-train --config-name base`

- add the baseline estimator
- Dlaczego nie robimy bezpośrednio konfig -> graf tylko dwa enkodery?
- investigate available graph embedders

1. Training pipeline (Mateusz)
2. Auto encoder for configuration (Michal)
3. Continuous training (Łukasz)

# Links

https://github.com/openai/CLIP -> trained OpenAI's CLIP
https://github.com/mlfoundations/open_clip?tab=readme-ov-file -> open implementation of CLIP
