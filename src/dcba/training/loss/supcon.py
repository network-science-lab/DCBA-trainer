"""Multi-positive supervised contrastive loss with embedding queue and NLS weighting."""

import logging

import torch
import torch.nn as nn
from torch import Tensor

_log = logging.getLogger(__name__)


def config_pairwise_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Euclidean distance between config vectors, as used for soft-negative weighting.

    Pulled out as a standalone function so that downstream analysis (e.g.
    ``scripts/embedding_stability.py``) can call the exact distance :class:`MultiPositiveSupConLoss`
    weights negatives by, instead of an independent reimplementation that could silently drift
    from it (e.g. if ``tau_dist`` or the metric itself changes here).

    :param x: ``(N, C)`` config vectors.
    :param y: ``(M, C)`` config vectors.

    :returns: ``(N, M)`` pairwise Euclidean distance matrix.
    """
    return torch.cdist(x, y, p=2)


class MultiPositiveSupConLoss(nn.Module):
    """
    Cross-modal Multi-Positive Supervised Contrastive Loss with soft-weighted negatives.

    Graph embeddings ``z_G`` act as anchors in the forward direction.  For anchor ``i`` the
    positive set contains the matching config embedding ``z_θ_i`` (cross-modal) and all other
    graph embeddings ``z_G_j`` that share the same ``instance_id`` label (same-modal).  All
    embeddings from instances with a different label are negatives, down-weighted by their
    distance in parameter space so that nearby configs are not treated as hard negatives.

    When ``bidirectional=True`` (default), the loss is run in both directions -- ``z_G`` anchors
    vs ``z_θ`` pool **and** ``z_θ`` anchors vs ``z_G`` pool -- and the two terms are summed.
    This CLIP-style symmetric objective aligns both embedding spaces from both sides, which is
    required for the test-time cross-modal decoding path ``z_G -> config_decoder -> θ_hat``.

    Both ``graph_embeddings`` and ``config_embeddings`` must have the same ``embedding_dim``.

    The loss is averaged over all ``(anchor, positive)`` pairs -- anchors with more in-batch
    positives contribute proportionally more signal, which is the desired behaviour when replica
    counts vary.  If the batch contains no valid positives at all a :class:`ValueError` is raised.

    :param temperature: Logit scale divisor ``τ``.
    :param tau_dist: Scale parameter for the negative soft-weight kernel
        ``exp(-||θ_i - θ_j|| / tau_dist)``.
    :param bidirectional: When ``True``, run the loss in both directions (CLIP-style) and sum.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        tau_dist: float = 1.0,
        bidirectional: bool = True,
    ) -> None:
        """Initialise with temperature, distance-kernel scale, and directionality."""
        super().__init__()
        self.temperature = temperature
        self.tau_dist = tau_dist
        self.bidirectional = bidirectional

    @staticmethod
    def _build_masks(
        anchor_labels: Tensor, all_labels: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """
        Build positive and self-exclusion masks over the ``(B_a, B_a + B_k)`` similarity grid.

        Columns correspond to ``[anchors; others]`` where ``B_k >= 0`` extra key entries may
        extend ``others`` beyond ``B_a``.  Anchor ``i`` is excluded from column ``i``
        (its own copy in the anchor portion of the pool).

        :param anchor_labels: ``(B_a,)`` integer group indices for the anchors.
        :param all_labels: ``(B_a + B_k,)`` group indices for the full key pool
            ``[anchors; others]``.
        :param device: Target device for the output tensors.

        :returns: Tuple ``(pos_mask, self_mask)`` each of shape ``(B_a, B_a + B_k)``, dtype bool.
        """
        b_a = anchor_labels.size(0)
        b_total = all_labels.size(0)

        self_mask = torch.zeros(b_a, b_total, dtype=torch.bool, device=device)
        self_mask[torch.arange(b_a, device=device), torch.arange(b_a, device=device)] = True

        pos_mask = anchor_labels.unsqueeze(1) == all_labels.unsqueeze(0)  # (B_a, B_a + B_k)
        pos_mask = pos_mask & ~self_mask

        return pos_mask, self_mask

    def _negative_weights(
        self,
        anchor_configs: Tensor,
        all_configs: Tensor,
        pos_mask: Tensor,
        self_mask: Tensor,
    ) -> Tensor:
        """
        Compute soft negative weights from pairwise config distances.

        ``w_ij = 1 - exp(-||θ_i - θ_j|| / tau_dist)`` for negative pairs, 0 elsewhere.

        :param anchor_configs: ``(B_a, C)`` config vectors for the anchors.
        :param all_configs: ``(B_a + B_k, C)`` config vectors for the full key pool.
        :param pos_mask: ``(B_a, B_a + B_k)`` bool mask of valid positives.
        :param self_mask: ``(B_a, B_a + B_k)`` bool mask of self-columns to exclude.

        :returns: ``(B_a, B_a + B_k)`` float weight matrix, zero on positive and self entries.
        """
        dist = config_pairwise_distance(anchor_configs, all_configs)  # (B_a, B_a + B_k)
        weights = 1.0 - torch.exp(-dist / self.tau_dist)
        neg_mask = ~pos_mask & ~self_mask
        return weights * neg_mask.float()

    @staticmethod
    def _compute_loss(sim: Tensor, pos_mask: Tensor, neg_weight: Tensor) -> Tensor:
        """
        Compute flat-pair-averaged contrastive loss given pre-scaled similarities.

        For each ``(anchor i, positive p)`` pair::

            loss_ip = -log( exp(sim_ip) / (Σ_{p'} exp(sim_ip') + Σ_n w_in * exp(sim_in)) )

        The result is averaged over all valid pairs.

        :param sim: ``(B, 2B)`` cosine similarities already divided by temperature.
        :param pos_mask: ``(B, 2B)`` bool mask identifying positive pairs.
        :param neg_weight: ``(B, 2B)`` float soft-weights for negative pairs (0 on positives).

        :returns: Scalar loss tensor.
        """
        sim_max = sim.detach().max(dim=1, keepdim=True).values  # for numerical stability
        exp_sim = torch.exp(sim - sim_max)  # subtract row-wise max; (B, 2B)

        pos_exp = exp_sim * pos_mask.float()  # (B, 2B)
        neg_exp = exp_sim * neg_weight  # (B, 2B)

        denom = pos_exp.sum(dim=1, keepdim=True) + neg_exp.sum(dim=1, keepdim=True)  # (B, 1)
        per_pair_loss = -torch.log(pos_exp / denom + 1e-8)  # (B, 2B)

        n_pairs = pos_mask.sum().clamp(min=1)
        return (per_pair_loss * pos_mask.float()).sum() / n_pairs

    def _directional_loss(
        self,
        anchors: Tensor,
        others: Tensor,
        anchor_labels: Tensor,
        key_labels: Tensor,
        anchor_configs: Tensor,
        key_configs: Tensor,
        precomputed_neg_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the contrastive loss for one direction: ``anchors`` vs ``[anchors; others]``.

        ``others`` may be larger than ``anchors`` (e.g. current-batch anchors against an
        extended key pool that includes memory-bank entries).  Both tensors must be L2-normalised.

        When ``precomputed_neg_weights`` is provided it is used directly (after masking out
        positives and self-pairs) instead of computing distances from ``anchor_configs`` and
        ``key_configs``.  This supports encoders where configs are not numeric vectors.

        :param anchors: ``(B_a, D)`` normalised anchor embeddings.
        :param others: ``(B_k, D)`` normalised key-pool embeddings; ``B_k >= B_a``.
        :param anchor_labels: ``(B_a,)`` group indices for the anchors.
        :param key_labels: ``(B_k,)`` group indices for the key pool.
        :param anchor_configs: ``(B_a, C)`` parameter vectors for the anchors.
        :param key_configs: ``(B_k, C)`` parameter vectors for the key pool.
        :param precomputed_neg_weights: Optional ``(B_a, B_a + B_k)`` raw weight matrix
            (before positive/self masking).  When supplied, replaces the internal
            :meth:`_negative_weights` call.

        :returns: Scalar loss tensor.
        """
        all_embeddings = torch.cat([anchors, others], dim=0)  # (B_a + B_k, D)
        all_labels = torch.cat([anchor_labels, key_labels], dim=0)  # (B_a + B_k,)
        all_configs = torch.cat([anchor_configs, key_configs], dim=0)  # (B_a + B_k, C)
        sim = torch.mm(anchors, all_embeddings.t()) / self.temperature  # (B_a, B_a + B_k)

        pos_mask, self_mask = self._build_masks(anchor_labels, all_labels, sim.device)
        if not pos_mask.any():
            _log.warning(
                "MultiPositiveSupConLoss: no valid positives found in this batch "
                "(all %d samples have distinct labels) -- returning zero loss.",
                anchor_labels.size(0),
            )
            return torch.tensor(0.0, device=anchors.device, dtype=anchors.dtype, requires_grad=True)
        if precomputed_neg_weights is not None:
            neg_mask = ~pos_mask & ~self_mask
            neg_weight = precomputed_neg_weights * neg_mask.float()
        else:
            neg_weight = self._negative_weights(anchor_configs, all_configs, pos_mask, self_mask)
        return self._compute_loss(sim, pos_mask, neg_weight)

    def forward(
        self,
        graph_embeddings: Tensor,
        config_embeddings: Tensor,
        labels: Tensor,
        configs: Tensor,
    ) -> Tensor:
        """
        Compute the multi-positive supervised contrastive loss.

        :param graph_embeddings: ``(B, D)`` L2-normalised graph embeddings ``z_G``; anchors in
            the forward direction.
        :param config_embeddings: ``(B, D)`` L2-normalised config embeddings ``z_θ``; anchors
            in the reverse direction when ``bidirectional=True``.
        :param labels: ``(B,)`` integer group index shared by both modalities.
        :param configs: ``(B, C)`` normalised parameter vectors for soft negative weighting.

        :returns: Scalar loss tensor with ``requires_grad=True``.
        """
        loss = self._directional_loss(
            graph_embeddings, config_embeddings, labels, labels, configs, configs
        )
        if self.bidirectional:
            loss = loss + self._directional_loss(
                config_embeddings, graph_embeddings, labels, labels, configs, configs
            )
        return loss
