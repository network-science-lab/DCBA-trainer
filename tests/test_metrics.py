"""Unit tests for dcba.training.metrics.retrieval."""

import torch

from dcba.training.metrics import retrieval_recall_at_k


class TestRetrievalRecallAtK:
    """Tests for retrieval_recall_at_k."""

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised rows of x."""
        return torch.nn.functional.normalize(x, dim=-1)

    def test_perfect_alignment_recall_is_one(self) -> None:
        """Recall@k == 1.0 when z_g and z_theta are identical."""
        z = self._normalise(torch.randn(8, 16))
        labels = torch.arange(8)
        recalls = retrieval_recall_at_k(z, z.clone(), labels)
        for k, v in recalls.items():
            assert v == 1.0, f"recall@{k} should be 1.0 for perfect alignment, got {v}"

    def test_all_misaligned_recall_is_zero(self) -> None:
        """Recall@1 == 0.0 when top-1 for every anchor is provably the wrong label.

        z_g[i] = e_i (i-th standard basis vector).
        z_theta[i] = e_{(i+1) % n} (shifted by one).
        Then z_g[i] @ z_theta[j] = 1 iff j == i-1 (mod n), so top-1 always
        retrieves z_theta with label (i-1) % n, never the correct label i.
        """
        n = 8
        z_g = torch.eye(n)
        z_theta = torch.roll(torch.eye(n), shifts=1, dims=0)
        labels = torch.arange(n)
        recalls = retrieval_recall_at_k(z_g, z_theta, labels, ks=(1,))
        assert recalls[1] == 0.0, f"recall@1 should be 0.0, got {recalls[1]}"

    def test_multi_positive_both_replicas_count_as_hit(self) -> None:
        """Both replicas of the same instance count as positives."""
        d = 16
        z = self._normalise(torch.randn(3, d))
        labels = torch.tensor([0, 0, 1])
        recalls = retrieval_recall_at_k(z, z.clone(), labels, ks=(1,))
        assert recalls[1] == 1.0

    def test_k_larger_than_pool_does_not_crash(self) -> None:
        """K values exceeding N are clamped without error."""
        z = self._normalise(torch.randn(3, 16))
        labels = torch.arange(3)
        recalls = retrieval_recall_at_k(z, z.clone(), labels, ks=(1, 5, 10, 100))
        assert set(recalls.keys()) == {1, 5, 10, 100}
        for v in recalls.values():
            assert 0.0 <= v <= 1.0

    def test_single_sample_recall_is_one(self) -> None:
        """Pool of size 1 always yields recall@1 == 1.0."""
        z = self._normalise(torch.randn(1, 16))
        labels = torch.tensor([42])
        recalls = retrieval_recall_at_k(z, z.clone(), labels, ks=(1,))
        assert recalls[1] == 1.0

    def test_recall_increases_with_k(self) -> None:
        """Recall@k is non-decreasing as k grows."""
        torch.manual_seed(0)
        n, d = 32, 64
        z_g = self._normalise(torch.randn(n, d))
        z_theta = self._normalise(torch.randn(n, d))
        labels = torch.arange(n)
        recalls = retrieval_recall_at_k(z_g, z_theta, labels, ks=(1, 5, 10))
        assert recalls[1] <= recalls[5] <= recalls[10]
