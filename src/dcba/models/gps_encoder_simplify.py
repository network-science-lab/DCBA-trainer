"""Simplified GPS-based graph encoder with a config-prediction head (no clustering pool)."""

import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor
from torch_geometric.nn import AttentionalAggregation, GINConv, GPSConv

from dcba.models.aggregators import LayerwiseAggregation
from dcba.models.types import ForwardOutput


class GPSEncoderSimplify(nn.Module):
    """GraphGPS-based graph encoder: maps a graph G to embedding ``z_g`` and predicts ``theta_hat``.

    A stripped-down variant of :class:`~dcba.models.gps_encoder.GPSEncoder` that drops the
    learned soft-clustering pooling branch entirely -- just the GPS message passing and a plain
    attention-weighted pool, with no auxiliary entropy losses.

    Combines local GIN message passing with global multi-head self-attention in each GPS layer,
    allowing the model to capture both local structural patterns and global graph statistics
    needed to recover entangled ABCD parameters (e.g. degree-distribution exponents and
    community-mixing parameter ``xi``).

    Each heterogeneous relation ``i`` is processed independently through all GPS layers, seeded
    from ``data["actor"].x[:, i:i+1]`` so that node features are semantically paired with the
    edges from the same community layer.  The resulting per-relation embeddings are aggregated
    with a trainable attention-weighted sum (``LayerwiseAggregation``).  Supports both
    ABCD (L=1, one relation) and mABCD (L>1, one relation per community layer).

    Expects ``data["actor"].x`` (node features from
    :class:`~dcba.dataset.transforms.CommunityToSize`) to be pre-computed before batching.

    :param hidden_dim: Channel width used throughout all GPS layers.
    :param num_layers: Number of GPS layers.
    :param num_heads: Number of multi-head attention heads in each GPS layer.
    :param embedding_dim: Dimensionality of the output graph embedding ``z_g``.
    :param output_dim: Dimensionality of the predicted config vector (e.g. 9 for ABCD).
    :param dropout: Dropout probability applied within GPS layers.
    :param attn_dropout: Dropout probability applied to attention weights.
    :param attn_type: Attention mechanism used in each GPS layer (e.g. ``"multihead"``,
        ``"performer"``). Use ``"performer"`` for O(N) linear attention when memory is limited.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        embedding_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        attn_type: str = "multihead",
    ) -> None:
        """Build GPS layers, pooling, and decoder from the supplied parameters."""
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        super().__init__()

        self._input_proj = nn.Linear(1, hidden_dim)

        self._gps_layers = nn.ModuleList(
            [
                GPSConv(
                    channels=hidden_dim,
                    conv=GINConv(
                        nn=nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                        ),
                        train_eps=True,
                    ),
                    heads=num_heads,
                    dropout=dropout,
                    attn_type=attn_type,
                    norm="layer_norm",
                    attn_kwargs={"dropout": attn_dropout},
                )
                for _ in range(num_layers)
            ]
        )

        self._aggregator = LayerwiseAggregation(hidden_dim)
        self._pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))

        self._proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, data: DCBAHeteroData) -> Tensor:
        """
        Compute graph embedding ``z_g`` from a batch of heterogeneous graphs.

        :param data: Batched heterogeneous graph data with ``data["actor"].x``
            pre-computed by the graph transform pipeline.

        :returns: Embedding tensor of shape ``(batch, embedding_dim)``.
        """
        batch_idx = data["actor"].batch

        y_relations = {}
        for i, (relation, edge_index) in enumerate(data.edge_index_dict.items()):
            h = self._input_proj(data["actor"].x[:, i : i + 1])
            for layer in self._gps_layers:
                h = layer(h, edge_index, batch=batch_idx)
            y_relations[relation] = h

        agg = self._aggregator(y_relations)
        pooled = self._pool(agg, index=batch_idx)

        return self._proj(pooled)

    def decode(self, z: Tensor) -> Tensor:
        """
        Predict config vector ``theta_hat`` from embedding ``z_g``.

        :param z: Embedding tensor of shape ``(batch, embedding_dim)``.

        :returns: Predicted config tensor of shape ``(batch, output_dim)``.
        """
        return self._decoder(z)

    def forward(self, batch: DCBAHeteroData) -> ForwardOutput:
        """
        Run the full graph-to-config forward pass.

        :param batch: Batched heterogeneous graph data.

        :returns: :class:`~dcba.models.types.ForwardOutput` with ``embedding`` set to ``z_g``
            and ``reconstruction`` set to ``theta_hat``.
        """
        z_g = self.encode(batch)
        theta_hat = self.decode(z_g)
        return ForwardOutput(embedding=z_g, reconstruction=theta_hat)
