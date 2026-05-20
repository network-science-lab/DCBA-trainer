"""Unit tests for DCBA loss functions."""

import torch
import torch.nn.functional as F

from dcba.training.loss import (
    ABCDConstraintPenaltyLoss,
    MultiPositiveSupConLoss,
)


def _make_batch(
    b: int,
    d: int,
    n_configs: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a synthetic batch for loss testing.

    Embeddings are L2-normalised to match the contract expected by
    :class:`~dcba.training.loss.MultiPositiveSupConLoss` (normalisation is the caller's
    responsibility, as in the training wrapper).

    :param b: Total batch size.
    :param d: Embedding dimensionality.
    :param n_configs: Number of distinct config groups; labels are assigned round-robin.
    :param device: Torch device string.

    :returns: Tuple ``(graph_embeddings, config_embeddings, labels, configs)``.
    """
    torch.manual_seed(0)
    graph_emb = F.normalize(torch.randn(b, d, device=device), dim=-1).requires_grad_(True)
    config_emb = F.normalize(torch.randn(b, d, device=device), dim=-1).requires_grad_(True)
    labels = torch.arange(b, device=device) % n_configs
    configs = torch.rand(b, 9, device=device)
    return graph_emb, config_emb, labels, configs


class TestBuildMasks:
    """Tests for :meth:`MultiPositiveSupConLoss._build_masks`."""

    @staticmethod
    def _symmetric_masks(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build masks for the symmetric (no-queue) case used in most tests."""
        all_labels = torch.cat([labels, labels])
        return MultiPositiveSupConLoss._build_masks(labels, all_labels, torch.device("cpu"))

    def test_self_column_excluded_from_positives(self) -> None:
        """Anchor i must not count column i (its own copy in anchor portion) as a positive."""
        b = 4
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, _ = self._symmetric_masks(labels)

        for i in range(b):
            assert not pos_mask[i, i], f"Anchor {i} should not be its own positive"

    def test_cross_modal_positive_included(self) -> None:
        """For each anchor i, column B+i (paired config embedding) must be a positive."""
        b = 4
        labels = torch.tensor([0, 1, 2, 3])  # all distinct -- only cross-modal positive each
        pos_mask, _ = self._symmetric_masks(labels)

        for i in range(b):
            assert pos_mask[i, b + i], f"Anchor {i}: paired config embedding must be positive"

    def test_same_modal_positives_included(self) -> None:
        """h_G_j with the same label as anchor i (j != i) must appear as positives."""
        labels = torch.tensor([0, 0, 0, 1])
        pos_mask, _ = self._symmetric_masks(labels)

        assert pos_mask[0, 1] and pos_mask[0, 2], "Anchor 0 missing same-modal positives"
        assert pos_mask[1, 0] and pos_mask[1, 2], "Anchor 1 missing same-modal positives"
        assert pos_mask[2, 0] and pos_mask[2, 1], "Anchor 2 missing same-modal positives"

    def test_different_label_not_positive(self) -> None:
        """Embeddings from a different label group must not appear as positives."""
        b = 4
        labels = torch.tensor([0, 0, 1, 1])
        pos_mask, _ = self._symmetric_masks(labels)

        for i in [0, 1]:
            for j in [2, 3, b + 2, b + 3]:
                assert not pos_mask[i, j], f"Anchor {i}, col {j}: cross-label must not be positive"

    def test_output_shapes(self) -> None:
        """pos_mask and self_mask must both be (B, 2B) booleans for the symmetric case."""
        b = 6
        labels = torch.zeros(b, dtype=torch.long)
        pos_mask, self_mask = self._symmetric_masks(labels)

        assert pos_mask.shape == (b, 2 * b)
        assert self_mask.shape == (b, 2 * b)
        assert pos_mask.dtype == torch.bool
        assert self_mask.dtype == torch.bool

    def test_asymmetric_key_pool(self) -> None:
        """With B_a anchors and B_k > B_a keys, masks are (B_a, B_a + B_k)."""
        b_a, b_k = 4, 6
        anchor_labels = torch.tensor([0, 1, 2, 3])
        extra_labels = torch.tensor([0, 1, 4, 5, 4, 5])
        all_labels = torch.cat([anchor_labels, extra_labels])
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(
            anchor_labels, all_labels, torch.device("cpu")
        )

        assert pos_mask.shape == (b_a, b_a + b_k)
        assert self_mask.shape == (b_a, b_a + b_k)
        # Anchor 0 (label 0) should match extra_labels[0] (label 0) at column b_a + 0
        assert pos_mask[0, b_a + 0], "Anchor 0 should match extra key at column b_a"


class TestNegativeWeights:
    """Tests for :meth:`MultiPositiveSupConLoss._negative_weights`."""

    @staticmethod
    def _sym_masks_and_configs(
        b: int, c: int, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (configs, all_configs, pos_mask, self_mask) for the symmetric (no-queue) case."""
        configs = torch.rand(b, c)
        all_configs = torch.cat([configs, configs])
        all_labels = torch.cat([labels, labels])
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(
            labels, all_labels, torch.device("cpu")
        )
        return configs, all_configs, pos_mask, self_mask

    def test_zero_on_positives_and_self(self) -> None:
        """Weight must be 0 on positive and self-column entries."""
        loss_fn = MultiPositiveSupConLoss()
        b, c = 4, 9
        labels = torch.tensor([0, 0, 1, 1])
        configs, all_configs, pos_mask, self_mask = self._sym_masks_and_configs(b, c, labels)

        weights = loss_fn._negative_weights(configs, all_configs, pos_mask, self_mask)

        assert (weights[pos_mask] == 0).all(), "Weights must be 0 on positive entries"
        assert (weights[self_mask] == 0).all(), "Weights must be 0 on self entries"

    def test_closer_configs_lower_weight(self) -> None:
        """Distant configs produce higher negative weight than identical configs."""
        loss_fn = MultiPositiveSupConLoss(tau_dist=1.0)
        b, c = 4, 9
        labels = torch.tensor([0, 0, 1, 1])
        _, _, pos_mask, self_mask = self._sym_masks_and_configs(b, c, labels)

        configs_close = torch.zeros(b, c)
        all_close = torch.cat([configs_close, configs_close])
        configs_far = torch.zeros(b, c)
        configs_far[2:] = 100.0
        all_far = torch.cat([configs_far, configs_far])

        w_close = loss_fn._negative_weights(configs_close, all_close, pos_mask, self_mask)
        w_far = loss_fn._negative_weights(configs_far, all_far, pos_mask, self_mask)

        assert w_far[0, 2].item() > w_close[0, 2].item(), (
            "Distant configs should produce higher negative weight than identical configs"
        )

    def test_output_shape(self) -> None:
        """Output must be ``(B, 2B)`` for the symmetric case."""
        loss_fn = MultiPositiveSupConLoss()
        b, c = 6, 9
        labels = torch.zeros(b, dtype=torch.long)
        configs, all_configs, pos_mask, self_mask = self._sym_masks_and_configs(b, c, labels)

        weights = loss_fn._negative_weights(configs, all_configs, pos_mask, self_mask)

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

    def test_unidirectional_flag(self) -> None:
        """Loss with ``bidirectional=False`` is a scalar with grad."""
        loss_fn = MultiPositiveSupConLoss(bidirectional=False)
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)
        loss = loss_fn(g, c, labels, configs)

        assert loss.ndim == 0
        assert loss.requires_grad

    def test_bidirectional_larger_than_unidirectional(self) -> None:
        """Bidirectional loss sums two directional terms, so it must exceed the unidirectional."""
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)
        loss_bi = MultiPositiveSupConLoss(bidirectional=True)(g, c, labels, configs)
        loss_uni = MultiPositiveSupConLoss(bidirectional=False)(g, c, labels, configs)

        assert loss_bi.item() > loss_uni.item()

    def test_bidirectional_backward_passes(self) -> None:
        """Gradients from the bidirectional loss flow to both graph and config embeddings."""
        loss_fn = MultiPositiveSupConLoss(bidirectional=True)
        g, c, labels, configs = _make_batch(b=8, d=16, n_configs=4)
        loss_fn(g, c, labels, configs).backward()

        assert g.grad is not None
        assert c.grad is not None


class TestDirectionalLoss:
    """Tests for :meth:`MultiPositiveSupConLoss._directional_loss` with precomputed weights."""

    @staticmethod
    def _make_directional_inputs(
        b: int, d: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(7)
        anchors = F.normalize(torch.randn(b, d), dim=-1)
        others = F.normalize(torch.randn(b, d), dim=-1)
        labels = torch.arange(b) % 2
        configs = torch.rand(b, 9)
        return anchors, others, labels, configs

    def test_precomputed_zero_weights_removes_negatives_from_denominator(self) -> None:
        """Zero precomputed weights eliminate negatives, making the loss differ from the default."""
        loss_fn = MultiPositiveSupConLoss()
        b, d = 4, 16
        anchors, others, labels, configs = self._make_directional_inputs(b, d)

        # All-zero weights: negatives never appear in the denominator
        zero_weights = torch.zeros(b, 2 * b)
        loss_zero_neg = loss_fn._directional_loss(
            anchors,
            others,
            labels,
            labels,
            configs,
            configs,
            precomputed_neg_weights=zero_weights,
        )
        loss_default = loss_fn._directional_loss(anchors, others, labels, labels, configs, configs)

        assert loss_zero_neg.ndim == 0
        assert not torch.isclose(loss_default, loss_zero_neg), (
            "Zero neg weights must produce a different loss than internally computed weights"
        )

    def test_precomputed_weights_matching_internal_gives_same_loss(self) -> None:
        """Supplying the internally computed weights must reproduce the default loss exactly."""
        loss_fn = MultiPositiveSupConLoss()
        b, d = 4, 16
        anchors, others, labels, configs = self._make_directional_inputs(b, d)

        all_labels = torch.cat([labels, labels])
        all_configs = torch.cat([configs, configs])
        pos_mask, self_mask = MultiPositiveSupConLoss._build_masks(
            labels, all_labels, torch.device("cpu")
        )
        internal_weights = loss_fn._negative_weights(configs, all_configs, pos_mask, self_mask)

        loss_precomputed = loss_fn._directional_loss(
            anchors,
            others,
            labels,
            labels,
            configs,
            configs,
            precomputed_neg_weights=internal_weights,
        )
        loss_default = loss_fn._directional_loss(anchors, others, labels, labels, configs, configs)

        assert torch.isclose(loss_precomputed, loss_default), (
            "Precomputed weights identical to internally computed must give the same loss"
        )

    def test_precomputed_weights_output_is_scalar_with_grad(self) -> None:
        """_directional_loss with precomputed_neg_weights returns a differentiable scalar."""
        loss_fn = MultiPositiveSupConLoss()
        b, d = 4, 16
        anchors, others, labels, configs = self._make_directional_inputs(b, d)
        anchors = anchors.requires_grad_(True)

        uniform_weights = torch.ones(b, 2 * b)
        loss = loss_fn._directional_loss(
            anchors,
            others,
            labels,
            labels,
            configs,
            configs,
            precomputed_neg_weights=uniform_weights,
        )
        loss.backward()

        assert loss.ndim == 0
        assert anchors.grad is not None


class TestABCDConstraintPenaltyLoss:
    """Tests for :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss`."""

    def _make_valid(self, b: int = 4) -> torch.Tensor:
        """Return a valid normalised config tensor in [0, 1] satisfying ordering constraints."""
        x = torch.zeros(b, 9)
        # n=0.5, t1=0.3, t2=0.3, xi=0.5, c_min=0.1, c_max=0.3, d_min=0.1, d_max=0.3, nout=0.2
        x[:] = torch.tensor([0.5, 0.3, 0.3, 0.5, 0.1, 0.3, 0.1, 0.3, 0.2])
        return x

    def test_output_is_scalar(self) -> None:
        """Loss must be a zero-dimensional scalar."""
        loss_fn = ABCDConstraintPenaltyLoss()
        x = self._make_valid()
        loss = loss_fn(x, x)
        assert loss.ndim == 0

    def test_no_violation_penalty_close_to_mse(self) -> None:
        """With no constraint violations the penalty term is zero, so loss equals MSE."""
        loss_fn = ABCDConstraintPenaltyLoss(lambda_penalty=1.0)
        x = self._make_valid()
        target = x.clone()
        target[:, 0] += 0.1
        loss = loss_fn(x, target)
        mse = F.mse_loss(x, target)
        assert abs(loss.item() - mse.item()) < 1e-6

    def test_violation_increases_loss(self) -> None:
        """A config that violates c_min <= c_max must produce a higher loss than a valid config."""
        loss_fn = ABCDConstraintPenaltyLoss(lambda_penalty=1.0)
        valid = self._make_valid()
        invalid = valid.clone()
        invalid[:, 4] = 0.9  # c_min > c_max (0.3)
        target = valid.clone()

        loss_valid = loss_fn(valid, target)
        loss_invalid = loss_fn(invalid, target)
        assert loss_invalid.item() > loss_valid.item()

    def test_backward_passes(self) -> None:
        """Gradients must flow back through the loss to x_hat."""
        loss_fn = ABCDConstraintPenaltyLoss()
        x = self._make_valid().requires_grad_(True)
        loss_fn(x, self._make_valid()).backward()
        assert x.grad is not None

    def test_lambda_penalty_scales_penalty(self) -> None:
        """Doubling lambda_penalty must produce a higher loss for a violating config."""
        x = self._make_valid()
        x[:, 4] = 0.9  # force a violation
        target = self._make_valid()
        loss_low = ABCDConstraintPenaltyLoss(lambda_penalty=0.1)(x, target)
        loss_high = ABCDConstraintPenaltyLoss(lambda_penalty=10.0)(x, target)
        assert loss_high.item() > loss_low.item()
