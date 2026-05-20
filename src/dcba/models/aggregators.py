"""Shared aggregation modules for graph encoders."""

import torch.nn as nn
from torch import Tensor, stack


class LayerwiseAggregation(nn.Module):
    """Trainable attention-weighted aggregation of per-relation (or per-layer) embeddings."""

    def __init__(self, hidden_channels: int) -> None:
        """Initialise the object."""
        super().__init__()
        self.attn = nn.Linear(hidden_channels, 1)

    def forward(self, h: dict[str, Tensor]) -> Tensor:
        """
        Trainable aggregation of per-relation embeddings.

        :param h: Per-relation embeddings dict ``{relation: [nb_actors, hidden_dim]}``.

        :returns: A tensor of shape ``[nb_actors, hidden_dim]``.
        """
        stacked = stack(list(h.values()))
        attn_scores = self.attn(stacked)
        attn_scores = nn.functional.softmax(attn_scores, dim=0)
        return (attn_scores * stacked).sum(dim=0)
