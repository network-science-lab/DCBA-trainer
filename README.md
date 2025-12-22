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
