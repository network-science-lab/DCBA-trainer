"""GPS-based graph encoder with a config-prediction head."""

import torch
import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor
from torch_geometric.nn import AttentionalAggregation, GINConv, GPSConv
from torch_geometric.utils import to_dense_batch

from dcba.models.aggregators import LayerwiseAggregation
from dcba.models.types import ForwardOutput


class GPSEncoder(nn.Module):
    """GraphGPS-based graph encoder: maps a graph G to embedding ``z_g`` and predicts ``theta_hat``.

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

    When ``num_clusters > 0``, a second, learned pooling branch runs alongside the existing
    flat attention pool: nodes get a soft assignment to up to ``num_clusters`` learned slots (no
    supervision from ground-truth community labels -- those are generator metadata unavailable
    on a real graph, see ``GNN_ENCODER_IDEAS.md`` idea 3), and quantiles of the resulting
    per-slot soft sizes are concatenated into the graph embedding. This is meant to give the
    model a route to distribution-shape graph properties (e.g. largest-community size, or the
    community-size power-law exponent) that a single attention-weighted average structurally
    struggles to represent -- two summary numbers (previously just ``[max, std]``) cannot pin
    down a heavy-tailed shape, so the full quantile set is exposed instead.

    ``num_clusters`` is a **budget, not an assumed community count** -- ABCD graphs vary in how
    many communities they actually have, so this must not hard-code a fixed number the way a
    plain k-means-style pool would. Two auxiliary entropy losses (summed into :attr:`aux_loss`
    after each :meth:`encode` call, ``None`` when ``num_clusters == 0``) push the assignment
    towards *discovering* how many slots a given graph actually needs, up to that budget, rather
    than always spreading mass over all of them: a per-node term (confident, near-one-hot
    assignment per node) and a per-graph usage term (concentrate total mass on a sparse subset of
    slots rather than spreading it evenly across all ``num_clusters``). This is the practical,
    batchable stand-in for HDBSCAN-style "don't fix K upfront" clustering -- an unbounded,
    density-based cluster count is not compatible with fixed-shape batched GPU tensors, so the
    budget is fixed but *usage* of it is learned per graph instead.

    :param hidden_dim: Channel width used throughout all GPS layers.
    :param num_layers: Number of GPS layers.
    :param num_heads: Number of multi-head attention heads in each GPS layer.
    :param embedding_dim: Dimensionality of the output graph embedding ``z_g``.
    :param output_dim: Dimensionality of the predicted config vector (e.g. 9 for ABCD).
    :param dropout: Dropout probability applied within GPS layers.
    :param attn_dropout: Dropout probability applied to attention weights.
    :param attn_type: Attention mechanism used in each GPS layer (e.g. ``"multihead"``,
        ``"performer"``). Use ``"performer"`` for O(N) linear attention when memory is limited.
    :param num_clusters: Upper bound on the number of soft-clustering slots for the hierarchical
        pooling branch described above -- a budget the model can under-use, not a claim about the
        true number of communities. ``0`` (default) disables the branch entirely, matching prior
        behaviour. Set generously relative to the largest expected community count; too tight a
        budget re-introduces the same "forced to squeeze everything into too few slots" problem
        this branch exists to avoid -- see ``scripts/recommend_num_clusters.py`` to pick a
        data-backed value instead of guessing.
    :param cluster_size_quantiles: Quantile levels (each in ``[0, 1]``) computed over the
        per-slot soft cluster sizes and concatenated into the graph embedding alongside the
        max-pooled cluster embedding. Only used when ``num_clusters > 0``. Include ``1.0`` to
        keep a plain max in the mix.
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
        num_clusters: int = 0,
        cluster_size_quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> None:
        """Build GPS layers, pooling, and decoder from the supplied parameters."""
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        super().__init__()

        self._num_clusters = num_clusters
        self._cluster_size_quantiles = tuple(cluster_size_quantiles)
        self.aux_loss: Tensor | None = None

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

        proj_in_dim = hidden_dim
        if num_clusters > 0:
            self._cluster_assign = nn.Linear(hidden_dim, num_clusters)
            # + cluster_pool (hidden_dim) + one channel per cluster-size quantile
            proj_in_dim += hidden_dim + len(self._cluster_size_quantiles)

        self._proj = nn.Sequential(
            nn.Linear(proj_in_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self._decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _cluster_pool(self, agg: Tensor, batch_idx: Tensor) -> Tensor:
        """
        Compute the learned soft-clustering pooling branch and set :attr:`aux_loss`.

        Uses a dense per-graph view (:func:`~torch_geometric.utils.to_dense_batch`) so the soft
        assignment and its reductions never mix nodes from different graphs in the batch.

        :param agg: ``(N, hidden_dim)`` post-GPS-layers node embeddings.
        :param batch_idx: ``(N,)`` graph index for each node.

        :returns: ``(batch, hidden_dim + len(cluster_size_quantiles))`` tensor: max-pooled cluster
            embedding concatenated with quantiles of the per-cluster soft sizes.
        """
        dense_agg, mask = to_dense_batch(agg, batch_idx)  # (B, N_max, H), (B, N_max)
        assign_logits = self._cluster_assign(dense_agg)  # (B, N_max, K)
        soft_assign = torch.softmax(assign_logits, dim=-1) * mask.unsqueeze(-1)

        # Entropy per node, masked mean over real (non-padding) nodes -- minimised so each node
        # commits to a small number of slots instead of spreading softly across all K.
        node_entropy = -(soft_assign * (soft_assign.clamp_min(1e-8)).log()).sum(dim=-1)
        node_entropy = (node_entropy * mask).sum() / mask.sum().clamp_min(1)

        cluster_embed = torch.einsum("bnk,bnh->bkh", soft_assign, dense_agg)  # (B, K, H)
        cluster_size = soft_assign.sum(dim=1)  # (B, K)

        # K is a budget, not an assumed community count (ABCD graphs vary in how many communities
        # they actually have) -- this term is what lets the model use fewer than K slots per
        # graph instead of always spreading mass over all of them. Minimising the entropy of the
        # *per-graph usage distribution* (how much total mass each of the K slots received,
        # normalised across slots) pushes usage towards a sparse subset of slots; unminimised, a
        # softmax over K categories has no pressure to leave any of them empty.
        usage = cluster_size / cluster_size.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        usage_entropy = -(usage * usage.clamp_min(1e-8).log()).sum(dim=-1).mean()

        self.aux_loss = node_entropy + usage_entropy

        cluster_pool = cluster_embed.max(dim=1).values  # (B, H)
        q = cluster_size.new_tensor(self._cluster_size_quantiles)  # (Q,)
        size_stats = torch.quantile(cluster_size, q, dim=1).T  # (B, Q)
        return torch.cat([cluster_pool, size_stats], dim=-1)

    def encode(self, data: DCBAHeteroData) -> Tensor:
        """
        Compute graph embedding ``z_g`` from a batch of heterogeneous graphs.

        Sets :attr:`aux_loss` as a side effect when ``num_clusters > 0`` (``None`` otherwise) --
        callers that use the clustering branch must add ``aux_loss`` to their training loss
        themselves, since this method's return type stays a plain embedding tensor for
        compatibility with other graph encoders (e.g. :class:`~dcba.models.gin_encoder.GINEncoder`).

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

        if self._num_clusters > 0:
            pooled = torch.cat([pooled, self._cluster_pool(agg, batch_idx)], dim=-1)

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
