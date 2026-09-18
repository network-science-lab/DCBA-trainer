"""GPS-based graph encoder whose clustering branch reads slot size *and* internal density."""

import math

import torch
import torch.nn as nn
from dcba_data_set.graph_io.data_models import DCBAHeteroData
from torch import Tensor
from torch_geometric.utils import scatter, to_dense_batch

from dcba.models.gps_encoder import GPSEncoder

#: Divisor that maps ``log1p(node count)`` into roughly ``[0, 1]`` for the graph-size readout
#: channel. Purely a conditioning constant -- unlike a scaler bound, nothing is clipped to it, so a
#: graph larger than this stays representable (the channel simply exceeds 1).
_LOG_SIZE_SCALE = math.log1p(10_000.0)

#: Floor used wherever a normaliser could otherwise be zero (empty slots, isolated nodes).
_EPS = 1e-8

#: Scalar channels :meth:`GPSEncoderShape._readout` appends after the two per-slot profiles:
#: four density extremes plus largest-slot exponent, graph size, budget occupancy and the cut.
_SCALAR_CHANNELS = 8


class GPSEncoderShape(GPSEncoder):
    """
    :class:`~dcba.models.gps_encoder.GPSEncoder` with a rebuilt cluster-branch readout.

    Everything else -- GPS layers, relation aggregation, attention pool, decoder, the soft
    assignment head, the two entropy aux terms -- is inherited unchanged, so the only code
    difference is how the clustering branch is *read out* into the graph embedding, plus one extra
    term in :attr:`aux_loss`. See ``.analysis/GNN_ENCODER_IDEAS.md`` for the
    measurements behind each channel, all taken on ground-truth communities (i.e. the best case a
    learned clustering could approach):

    - **Size profile, scale-free and occupancy-invariant** (:meth:`_readout`). ``GPSEncoder``
      concatenates plain quantiles of the raw ``(num_clusters,)`` soft size vector: values that
      scale with ``n`` and reach ``1e4``, fed into the projection next to unit-scale pooled
      features, and -- with ``num_clusters`` set generously above the true community count, as it
      must be -- mostly reading empty slots rather than the shape of the size distribution. Here
      the quantiles are read off the *size-biased* distribution ("how large is the slot of a
      randomly chosen node"), so empty slots drop out entirely, and reported as ratios to the
      largest slot, which is exactly invariant to graph size.
    - **Density channels**, the share of each slot's edge volume that stays inside it: a profile
      read at the same slot positions as the size profile, plus four *extremes* (least and most
      dense occupied slot, and the least dense slot's node mass and size). Size cannot separate the
      ABCD outlier set from a small legitimate community, so ``nout`` is invisible to a size-only
      readout; outliers are precisely the group that keeps almost none of its volume internally.
      The extremes are what carry it -- the size-biased profile is dominated by the large slots and
      leaves ``nout`` at oracle R2 ``0.03``, while adding the extremes reaches ``0.89``.
    - **Normalised-cut channel**, the graph-level share of edge volume kept inside slots.
      ``1 - cut`` is essentially the definition of ``xi``: used raw as a prediction of ``xi``, with
      no fitting whatsoever, it scores R2 ``0.91``, and the full readout takes ``xi`` from oracle
      R2 ``-0.33`` (size profile alone) to ``0.94``. ``GPSEncoder`` never exposed it.
    - **A soft modularity term** (``modularity_weight``). ``GPSEncoder``'s assignment head reads
      only ``h``, so nothing gives its slots a reason to be densely connected groups -- a
      precondition for any of the statistics above to mean what their names say. Modularity is used
      rather than MinCutPool's normalised cut (Bianchi et al. 2020) because the latter is maximised
      by collapsing every node into one slot and therefore needs an orthogonality counterweight,
      which in turn pushes towards using every slot -- and that fragmentation regime is measurably
      the more destructive of the two. Modularity's optimum is the true partition itself, so one
      term and one weight suffice; see :meth:`_modularity` for the measurements.

    Note the per-slot statistics cost one sparse ``(N, N) x (N, K)`` product per forward pass,
    roughly one GIN layer's worth of work, and are computed whether or not ``modularity_weight`` is
    set, since the readout channels depend on them.

    :param hidden_dim: Channel width used throughout all GPS layers.
    :param num_layers: Number of GPS layers.
    :param num_heads: Number of multi-head attention heads in each GPS layer.
    :param embedding_dim: Dimensionality of the output graph embedding ``z_g``.
    :param output_dim: Dimensionality of the predicted config vector (e.g. 9 for ABCD).
    :param dropout: Dropout probability applied within GPS layers.
    :param attn_dropout: Dropout probability applied to attention weights.
    :param attn_type: Attention mechanism used in each GPS layer (e.g. ``"multihead"``,
        ``"performer"``). Use ``"performer"`` for O(N) linear attention when memory is limited.
    :param num_clusters: Upper bound on the number of soft-clustering slots (a budget, not an
        assumed community count). ``0`` disables the branch entirely. Set above the largest expected
        community count, but note ``(batch, N_max, num_clusters)`` dominates this branch's memory;
        see ``scripts/recommend_num_clusters.py`` to pick a data-backed value.
    :param cluster_size_quantiles: Quantile levels (each in ``[0, 1]``) of the size-biased cluster
        size distribution, at which both the size and the density profile are read. ``0.0`` reads
        the largest slot, ``1.0`` the smallest.
    :param usage_entropy_weight: Weight on the per-graph usage-entropy term inside
        :attr:`aux_loss`. Its minimum is all mass on a single slot, which collapses the cluster
        sizes and destroys the size-distribution signal the readout exposes, so ``0.0`` is the
        sensible value whenever the readout is in use.
    :param node_entropy_weight: Weight on the inherited per-node entropy term, which pushes each
        node towards a one-hot assignment. ``1.0`` reproduces
        :class:`~dcba.models.gps_encoder.GPSEncoder`, but it fights the modularity term and wins.
        Its range is ``[0, log(num_clusters)]`` -- 4.16 at ``num_clusters=64`` -- against
        modularity's
        ``[0, 1]``, in practice ``[0, 0.41]``, so it outweighs it roughly 10 to 1. Observed on run
        ``z7pulj4r``: over 140 steps the effective slot count fell 56 to 1.8 out of 64, the cut
        climbed towards 1.0 and modularity went *negative* -- the whole branch collapsed into one
        slot and every readout channel became a constant. Set ``0.0`` when ``modularity_weight`` is
        doing the shaping: sharp assignments are not the goal, and modularity already penalises the
        uniform solution too (a uniform assignment scores ``Q = 0``, exactly like collapse).
    :param modularity_weight: Weight on the negated soft modularity term inside :attr:`aux_loss`,
        the only thing tying the learned slots to the graph's edges. ``0.0`` disables it, leaving
        the assignment head reading ``h`` alone -- in which case the density and cut readout
        channels have no reason to describe communities and the supervised loss is free to find a
        shortcut that fits them without any community structure. Needs no counterweight, unlike the
        MinCutPool pair it replaces; see :meth:`_modularity`. Watch :attr:`modularity` (ground-truth
        ABCD partitions score ~0.41) and :attr:`usage_entropy` (``exp(value)`` ~ occupied slots).
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
        usage_entropy_weight: float = 1.0,
        node_entropy_weight: float = 1.0,
        modularity_weight: float = 0.0,
    ) -> None:
        """Build the inherited encoder, then widen the projection for the richer readout."""
        super().__init__(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
            attn_type=attn_type,
            num_clusters=num_clusters,
            cluster_size_quantiles=cluster_size_quantiles,
            usage_entropy_weight=usage_entropy_weight,
        )
        self._node_entropy_weight = node_entropy_weight
        self._modularity_weight = modularity_weight
        #: Detached normalised cut of the last :meth:`encode` call, in ``[0, 1]``, logged as a
        #: diagnostic: the fraction of edge volume that stays inside a slot. ``1.0`` means a single
        #: slot swallowed the graph; ``1 - value`` tracks ``xi``.
        self.normalised_cut: Tensor | None = None
        #: Detached soft modularity of the last :meth:`encode` call, logged as a diagnostic
        #: (independent of ``modularity_weight``). Ground-truth ABCD communities score ~0.41 on
        #: average and ~0.61 where ``xi < 0.5``; ~0.0 means the assignment carries no community
        #: structure, so the density and cut readout channels are meaningless for that batch.
        self.modularity: Tensor | None = None

        if num_clusters > 0:
            # The parent sized _proj for a single Q-wide raw-size readout; this one emits a size
            # profile and a density profile plus _SCALAR_CHANNELS scalars.
            self._proj = nn.Sequential(
                nn.Linear(
                    2 * hidden_dim + 2 * len(self._cluster_size_quantiles) + _SCALAR_CHANNELS,
                    embedding_dim,
                ),
                nn.LayerNorm(embedding_dim),
            )

    def _slot_statistics(
        self, soft_assign: Tensor, edge_index: Tensor, batch_idx: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute per-graph, per-slot soft size, intra-slot edge volume, and total edge volume.

        Evaluated on the batch's sparse adjacency rather than a dense one: batched graphs occupy
        disjoint node index ranges, so a single sparse matrix product is already block-diagonal per
        graph, and a dense ``(B, N_max, N_max)`` adjacency would be far too large at these graph
        sizes. ``torch.sparse.mm`` also avoids ever materialising an ``(E, K)`` intermediate.

        :param soft_assign: ``(N, K)`` per-node soft assignment, rows summing to 1.
        :param edge_index: ``(2, E)`` edge index of the batch.
        :param batch_idx: ``(N,)`` graph index for each node.

        :returns: ``(size, intra_volume, volume)``, each ``(B, K)``.
        """
        num_nodes = soft_assign.size(0)
        adjacency = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=soft_assign.device, dtype=soft_assign.dtype),
            (num_nodes, num_nodes),
        )
        neighbour_mass = torch.sparse.mm(adjacency, soft_assign)  # (N, K)
        degree = scatter(
            torch.ones_like(edge_index[0], dtype=soft_assign.dtype),
            edge_index[0],
            dim=0,
            dim_size=num_nodes,
        )

        size = scatter(soft_assign, batch_idx, dim=0)
        intra_volume = scatter(soft_assign * neighbour_mass, batch_idx, dim=0)
        volume = scatter(degree.unsqueeze(-1) * soft_assign, batch_idx, dim=0)
        return size, intra_volume, volume

    def _readout(self, size: Tensor, intra_volume: Tensor, volume: Tensor) -> Tensor:
        """
        Summarise the slots as a scale-free size profile, a density profile, and four scalars.

        Both profiles are read at the quantiles of the **size-biased** size distribution -- the slot
        a uniformly random *node* belongs to -- rather than of the raw ``(B, K)`` vector. Slots that
        hold no nodes contribute nothing to that distribution, so the profiles do not change when
        the ``num_clusters`` budget is set generously above the true community count (reading
        raw-vector quantiles with ``K`` far above the real count returns mostly zeros, i.e. it
        measures how much of the budget went unused rather than the shape of the distribution).

        The size profile is reported as plain ratios to the largest slot, which is exactly invariant
        to multiplying every size by a constant, so it carries shape and nothing else -- note a
        log-ratio such as ``log(size) / log(largest)`` would *not* be: it is an exponent, and
        exponents move when the scale does. Absolute scale gets its own channels instead.

        :param size: ``(B, K)`` soft node count per slot.
        :param intra_volume: ``(B, K)`` edge volume kept inside each slot.
        :param volume: ``(B, K)`` total edge volume incident to each slot.

        :returns: ``(B, 2 * len(cluster_size_quantiles) + 8)`` tensor: the size profile, the density
            profile, the four density extremes, then the largest slot as an exponent of the node
            count, the node count itself, the fraction of the slot budget occupied, and the graph's
            normalised cut.
        """
        total = size.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B, 1) ~ node count
        order = size.argsort(dim=-1, descending=True)
        sorted_size = size.gather(-1, order)
        density = (intra_volume / volume.clamp_min(_EPS)).gather(-1, order)  # (B, K), in [0, 1]
        largest = sorted_size[:, :1]

        # Fraction of nodes living in the k largest slots: non-decreasing, reaching 1 at k = K.
        node_fraction = (sorted_size.cumsum(dim=-1) / total).contiguous()  # (B, K)
        levels = size.new_tensor(self._cluster_size_quantiles)  # (Q,)
        levels = levels.expand(size.size(0), -1).contiguous()  # (B, Q)
        picked = torch.searchsorted(node_fraction, levels).clamp(max=size.size(-1) - 1)

        size_profile = sorted_size.gather(-1, picked) / largest.clamp_min(_EPS)  # (B, Q)
        density_profile = density.gather(-1, picked)  # (B, Q)

        # Density *extremes*, not just the profile above. The profile is read at size-biased
        # positions, so it is dominated by the large slots and misses the ABCD outlier set, which is
        # small; measured on ground-truth partitions the profile alone recovers nout/n at R2 0.03
        # while adding these four channels takes it to 0.89. A slot holding less than one node's
        # worth of mass is empty, and an empty slot has zero volume and therefore zero density --
        # it would win every argmin -- so the extremes are taken over occupied slots only.
        occupied = sorted_size >= 1.0
        for_min = torch.where(occupied, density, torch.ones_like(density))
        least_dense = for_min.argmin(dim=-1, keepdim=True)  # (B, 1)
        density_extremes = torch.cat(
            [
                for_min.gather(-1, least_dense),
                torch.where(occupied, density, torch.zeros_like(density)).amax(-1, keepdim=True),
                sorted_size.gather(-1, least_dense) / total,
                sorted_size.gather(-1, least_dense) / largest.clamp_min(_EPS),
            ],
            dim=-1,
        )  # (B, 4)

        largest_exponent = torch.log1p(largest) / torch.log1p(total)  # (B, 1)
        size_scale = torch.log1p(total) / _LOG_SIZE_SCALE  # (B, 1)

        share = size / total
        entropy = -(share * share.clamp_min(_EPS).log()).sum(dim=-1, keepdim=True)
        occupancy = entropy.exp() / size.size(-1)  # (B, 1), in [0, 1]

        return torch.cat(
            [
                size_profile,
                density_profile,
                density_extremes,
                largest_exponent,
                size_scale,
                occupancy,
                self._normalised_cut(intra_volume, volume),
            ],
            dim=-1,
        )

    @staticmethod
    def _normalised_cut(intra_volume: Tensor, volume: Tensor) -> Tensor:
        """
        Compute the share of each graph's edge volume that stays inside a slot.

        Shared by :meth:`_readout`, which exposes it as an embedding channel, and
        :meth:`_shape_pool`, which optionally weights it into :attr:`aux_loss` -- deliberately
        recomputed rather than passed through :attr:`normalised_cut`, which is detached and would
        silently contribute no gradient to any loss reading it back.

        :param intra_volume: ``(B, K)`` edge volume kept inside each slot.
        :param volume: ``(B, K)`` total edge volume incident to each slot.

        :returns: ``(B, 1)`` normalised cut, in ``[0, 1]``. ``1 - value`` tracks ``xi``.
        """
        return intra_volume.sum(dim=-1, keepdim=True) / volume.sum(dim=-1, keepdim=True).clamp_min(
            _EPS
        )

    @staticmethod
    def _modularity(intra_volume: Tensor, volume: Tensor) -> Tensor:
        """
        Compute soft Newman modularity per graph, the term that ties the slots to the edges.

        :param intra_volume: ``(B, K)`` edge volume kept inside each slot.
        :param volume: ``(B, K)`` total edge volume incident to each slot.

        :returns: ``(B,)`` modularity, in ``[-0.5, 1]``. Higher is a better community partition.
        """
        two_m = volume.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        return (intra_volume / two_m - (volume / two_m) ** 2).sum(dim=-1)

    def _shape_pool(self, agg: Tensor, batch_idx: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute the clustering branch and set :attr:`aux_loss`, replacing ``_cluster_pool``.

        The assignment is computed on the flat ``(N, K)`` node tensor and only then viewed per graph
        (:func:`~torch_geometric.utils.to_dense_batch`), so no reduction ever mixes nodes from
        different graphs in the batch and padding rows never reach the softmax.

        :param agg: ``(N, hidden_dim)`` post-GPS-layers node embeddings.
        :param batch_idx: ``(N,)`` graph index for each node.
        :param edge_index: ``(2, E)`` edge index of the batch.

        :returns: ``(batch, hidden_dim + 2 * len(cluster_size_quantiles) + 8)`` tensor: max-pooled
            cluster embedding concatenated with :meth:`_readout`'s channels.
        """
        soft_assign = torch.softmax(self._cluster_assign(agg), dim=-1)  # (N, K)
        dense_assign, _ = to_dense_batch(soft_assign, batch_idx)  # (B, N_max, K)
        dense_agg, _ = to_dense_batch(agg, batch_idx)  # (B, N_max, H)

        # Entropy per node, mean over real nodes -- minimised so each node commits to a small
        # number of slots instead of spreading softly across all K. No mask is needed, unlike
        # GPSEncoder's version: that one computes on the padded (B, N_max, K) view and has to
        # exclude padding rows, whereas soft_assign here is the flat (N, K) tensor of real nodes
        # only, so a plain mean is exactly its masked mean.
        node_entropy = -(soft_assign * soft_assign.clamp_min(_EPS).log()).sum(dim=-1).mean()

        size, intra_volume, volume = self._slot_statistics(soft_assign, edge_index, batch_idx)

        # Per-graph usage entropy over the K slots -- minimising it lets the model concentrate on
        # a sparse subset of slots rather than using all K. Its minimum is a single occupied slot,
        # so weight 0 disables it when the readout needs spread-out sizes.
        usage = size / size.sum(dim=-1, keepdim=True).clamp_min(_EPS)
        usage_entropy = -(usage * usage.clamp_min(_EPS).log()).sum(dim=-1).mean()

        cut = self._normalised_cut(intra_volume, volume)
        self.usage_entropy = usage_entropy.detach()
        self.normalised_cut = cut.mean().detach()

        modularity = self._modularity(intra_volume, volume)
        self.modularity = modularity.mean().detach()

        # Modularity enters negated, so minimising the loss maximises it -- i.e. drives the slots
        # towards densely connected groups. No counterweight term: unlike a normalised cut, its
        # optimum is neither one slot nor all of them (see _modularity).
        self.aux_loss = (
            self._node_entropy_weight * node_entropy
            + self._usage_entropy_weight * usage_entropy
            - self._modularity_weight * modularity.mean()
        )

        cluster_embed = torch.einsum("bnk,bnh->bkh", dense_assign, dense_agg)  # (B, K, H)
        cluster_pool = cluster_embed.max(dim=1).values  # (B, H)
        return torch.cat([cluster_pool, self._readout(size, intra_volume, volume)], dim=-1)

    def encode(self, data: DCBAHeteroData) -> Tensor:
        """
        Compute graph embedding ``z_g``, routing the clustering branch through :meth:`_shape_pool`.

        Identical to :meth:`~dcba.models.gps_encoder.GPSEncoder.encode` except that the branch also
        receives the batch's edges, which the density and cut channels need.

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
            # The branch reads the relation-fused node embeddings, so its density and cut channels
            # are evaluated on the union of every relation's edges (a single relation for ABCD, one
            # per community layer for mABCD).
            edges = torch.cat(list(data.edge_index_dict.values()), dim=-1)
            pooled = torch.cat([pooled, self._shape_pool(agg, batch_idx, edges)], dim=-1)

        return self._proj(pooled)
