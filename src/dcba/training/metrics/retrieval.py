"""Cross-modal retrieval recall metrics for contrastive embedding evaluation."""

import torch
from torch import Tensor


def retrieval_recall_at_k(
    z_g: Tensor,
    z_theta: Tensor,
    labels: Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[int, float]:
    """
    Compute cross-modal retrieval recall at k for each k in ``ks``.

    For every graph embedding ``z_g_i``, retrieves the top-k config embeddings by
    cosine similarity.  A hit occurs when at least one of those k entries shares the
    same ``instance_id`` label as ``z_g_i`` (i.e. encodes the same ABCD/mABCD config).
    Multiple ``z_theta`` entries can be positive for the same anchor when the dataset
    contains several replicas of the same instance.

    All inputs must be on CPU.  ``z_g`` and ``z_theta`` must already be L2-normalised
    so that dot product equals cosine similarity.

    :param z_g: ``(N, D)`` L2-normalised graph embeddings.
    :param z_theta: ``(N, D)`` L2-normalised config embeddings.
    :param labels: ``(N,)`` integer instance labels; entries with the same label are
        treated as positives for each other.
    :param ks: Tuple of k values for which to compute recall.

    :returns: Dict mapping each k to the fraction of anchors (in [0, 1]) for which at
        least one positive appeared in the top-k retrieved config embeddings.
    """
    n = z_g.size(0)
    sim = z_g @ z_theta.t()  # (N, N) cosine similarities

    max_k = min(max(ks), n)
    topk_indices = sim.topk(max_k, dim=1).indices  # (N, max_k)

    pos_mask = labels.unsqueeze(1) == labels.unsqueeze(0)  # (N, N) ground-truth positives

    recalls: dict[int, float] = {}
    for k in ks:
        effective_k = min(k, n)
        gathered = torch.zeros(n, n, dtype=torch.bool)
        gathered.scatter_(1, topk_indices[:, :effective_k], True)
        hit = (gathered & pos_mask).any(dim=1)  # (N,) -- at least one positive in top-k
        recalls[k] = hit.float().mean().item()

    return recalls
