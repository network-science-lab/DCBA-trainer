"""Unit tests for DCBA loss functions."""

import torch

from dcba.training.loss import MultiPositiveSupConLoss


def _make_batch(
    b: int,
    d: int,
    n_configs: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a synthetic batch for loss testing.

    :param b: Total batch size.
    :param d: Embedding dimensionality.
    :param n_configs: Number of distinct config groups; labels are assigned round-robin.
    :param device: Torch device string.

    :returns: Tuple ``(graph_embeddings, config_embeddings, labels, configs)``.
    """
    torch.manual_seed(0)
    graph_emb = torch.randn(b, d, device=device, requires_grad=True)
    config_emb = torch.randn(b, d, device=device, requires_grad=True)
    labels = torch.arange(b, device=device) % n_configs
    configs = torch.rand(b, 9, device=device)
    return graph_emb, config_emb, labels, configs


class TestBuildMasks:
    """Tests for :meth:`MultiPositiveSupConLoss._build_masks`."""

    def test_self_column_excluded_from_positives(self) -> None:
        """Anchor i must not count column i (its own h_G copy) as a positive."""
        b = 4
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, _ = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        for i in range(b):
            assert not pos_mask[i, i], f"Anchor {i} should not be its own positive"

    def test_cross_modal_positive_included(self) -> None:
        """For each anchor i, column B+i (paired h_θ_i) must be a positive."""
        b = 4
        labels = torch.tensor([0, 1, 2, 3])  # all distinct — only cross-modal positive each
        pos_mask, _ = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        for i in range(b):
            assert pos_mask[i, b + i], f"Anchor {i}: paired config embedding must be positive"

    def test_same_modal_positives_included(self) -> None:
        """h_G_j with the same label as anchor i (j != i) must appear as positives."""
        b = 4
        labels = torch.tensor([0, 0, 0, 1])
        pos_mask, _ = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        # Anchors 0, 1, 2 share label 0 — each should see the other two as same-modal positives
        assert pos_mask[0, 1] and pos_mask[0, 2], "Anchor 0 missing same-modal positives"
        assert pos_mask[1, 0] and pos_mask[1, 2], "Anchor 1 missing same-modal positives"
        assert pos_mask[2, 0] and pos_mask[2, 1], "Anchor 2 missing same-modal positives"

    def test_different_label_not_positive(self) -> None:
        """Embeddings from a different label group must not appear as positives."""
        b = 4
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, _ = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        # Anchors 0/1 (label 0) must not treat columns 2/3 or B+2/B+3 (label 1) as positives
        for i in [0, 1]:
            for j in [2, 3, b + 2, b + 3]:
                assert not pos_mask[i, j], f"Anchor {i}, col {j}: cross-label must not be positive"

    def test_output_shapes(self) -> None:
        """pos_mask and self_mask must both be (B, 2B) booleans."""
        b = 6
        labels = torch.zeros(b, dtype=torch.long)
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        assert pos_mask.shape == (b, 2 * b)
        assert self_mask.shape == (b, 2 * b)
        assert pos_mask.dtype == torch.bool
        assert self_mask.dtype == torch.bool


class TestNegativeWeights:
    """Tests for :meth:`MultiPositiveSupConLoss._negative_weights`."""

    def test_zero_on_positives_and_self(self) -> None:
        """Weight must be 0 on positive and self-column entries."""
        loss_fn = MultiPositiveSupConLoss()
        b, c = 4, 9
        configs = torch.rand(b, c)
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        weights = loss_fn._negative_weights(configs, pos_mask, self_mask)

        assert (weights[pos_mask] == 0).all(), "Weights must be 0 on positive entries"
        assert (weights[self_mask] == 0).all(), "Weights must be 0 on self entries"

    def test_closer_configs_higher_weight(self) -> None:
        """Identical configs produce higher negative weight than distant configs."""
        loss_fn = MultiPositiveSupConLoss(tau_dist=1.0)
        b, c = 4, 9
        # Two groups: labels 0 (items 0-1) and 1 (items 2-3)
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        configs_close = torch.zeros(b, c)  # all configs identical
        configs_far = torch.zeros(b, c)
        configs_far[2:] = 100.0  # group 1 very far from group 0

        w_close = loss_fn._negative_weights(configs_close, pos_mask, self_mask)
        w_far = loss_fn._negative_weights(configs_far, pos_mask, self_mask)

        # Anchor 0 vs column 2 (negative): close-config weight should exceed far-config weight
        assert w_close[0, 2].item() > w_far[0, 2].item(), (
            "Identical configs should produce higher negative weight than distant configs"
        )

    def test_output_shape(self) -> None:
        """Output must be ``(B, 2B)``."""
        loss_fn = MultiPositiveSupConLoss()
        b, c = 6, 9
        configs = torch.rand(b, c)
        labels = torch.zeros(b, dtype=torch.long)
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(labels, b, torch.device("cpu"))

        weights = loss_fn._negative_weights(configs, pos_mask, self_mask)

        assert weights.shape == (b, 2 * b)


class TestComputeLoss:
    """Tests for :meth:`MultiPositiveSupConLoss._compute_loss`."""

    def test_non_negative_scalar(self) -> None:
        """Loss must be a non-negative scalar."""
        b, size = 4, 2 * 4
        torch.manual_seed(1)
        sim = torch.randn(b, size)
        pos_mask = torch.zeros(b, size, dtype=torch.bool)
        pos_mask[torch.arange(b), torch.arange(b) + b] = True
        neg_weight = torch.rand(b, size) * (~pos_mask).float()

        loss = MultiPositiveSupConLoss._compute_loss(sim, pos_mask, neg_weight)

        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_perfect_alignment_lower_than_random(self) -> None:
        """Loss should be lower when positives score much higher than all negatives."""
        b = 4
        pos_mask = torch.zeros(b, 2 * b, dtype=torch.bool)
        pos_mask[torch.arange(b), torch.arange(b) + b] = True
        # Uniform negative weights on all non-positive, non-self entries
        self_mask = torch.zeros(b, 2 * b, dtype=torch.bool)
        self_mask[torch.arange(b), torch.arange(b)] = True
        neg_weight = (~pos_mask & ~self_mask).float()

        # Perfect: positives score +10, all negatives score -10
        sim_perfect = torch.full((b, 2 * b), -10.0)
        sim_perfect[torch.arange(b), torch.arange(b) + b] = 10.0

        # Random similarities
        torch.manual_seed(2)
        sim_random = torch.randn(b, 2 * b)

        loss_perfect = MultiPositiveSupConLoss._compute_loss(sim_perfect, pos_mask, neg_weight)
        loss_random = MultiPositiveSupConLoss._compute_loss(sim_random, pos_mask, neg_weight)

        assert loss_perfect.item() < loss_random.item(), (
            "Perfect-alignment loss should be lower than random-similarity loss"
        )


class TestMultiPositiveSupConLossForward:
    """End-to-end tests for :class:`~dcba.training.loss.MultiPositiveSupConLoss`."""

    def test_output_is_scalar_with_grad(self) -> None:
        """Loss is a zero-dimensional tensor with requires_grad."""
        loss_fn = MultiPositiveSupConLoss()
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)
        loss = loss_fn(g, c, labels, configs)

        assert loss.ndim == 0
        assert loss.requires_grad

    def test_output_is_non_negative(self) -> None:
        """Loss value is ≥ 0."""
        loss_fn = MultiPositiveSupConLoss()
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)

        assert loss_fn(g, c, labels, configs).item() >= 0.0

    def test_all_same_label_no_negatives(self) -> None:
        """When all samples share one label the loss is well-defined with no hard negatives."""
        loss_fn = MultiPositiveSupConLoss()
        b, d = 6, 16
        g, c, _, configs = _make_batch(b=b, d=d, n_configs=1)
        labels = torch.zeros(b, dtype=torch.long)

        loss = loss_fn(g, c, labels, configs)

        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_all_distinct_labels_still_valid(self) -> None:
        """With distinct labels each anchor still has its cross-modal positive — loss is finite."""
        loss_fn = MultiPositiveSupConLoss()
        b, d = 6, 16
        g, c, _, configs = _make_batch(b=b, d=d, n_configs=b)
        labels = torch.arange(b)

        loss = loss_fn(g, c, labels, configs)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_backward_passes(self) -> None:
        """Gradients flow back to both graph and config embeddings."""
        loss_fn = MultiPositiveSupConLoss()
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)
        loss_fn(g, c, labels, configs).backward()

        assert g.grad is not None
        assert c.grad is not None
