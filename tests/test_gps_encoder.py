"""Unit tests for GPSEncoder's soft-clustering pooling branch."""

import torch

from dcba.models.gps_encoder import GPSEncoder


def _make_encoder(usage_entropy_weight: float) -> GPSEncoder:
    """Build a small encoder with the clustering branch enabled."""
    return GPSEncoder(
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=8,
        output_dim=9,
        num_clusters=6,
        usage_entropy_weight=usage_entropy_weight,
    )


def _synthetic_nodes(n_nodes: int = 30, hidden_dim: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return post-GPS node embeddings and a two-graph batch index."""
    torch.manual_seed(0)
    agg = torch.randn(n_nodes, hidden_dim)
    batch_idx = torch.cat([torch.zeros(n_nodes // 2), torch.ones(n_nodes - n_nodes // 2)]).long()
    return agg, batch_idx


class TestUsageEntropyWeight:
    """aux_loss composition must follow the usage_entropy_weight parameter."""

    def test_zero_weight_drops_usage_term(self) -> None:
        """With weight 0 the aux loss is strictly smaller than with weight 1 on the same input."""
        agg, batch_idx = _synthetic_nodes()
        enc_off, enc_on = _make_encoder(0.0), _make_encoder(1.0)
        enc_on.load_state_dict(enc_off.state_dict())  # identical parameters, same assignments

        enc_off._cluster_pool(agg, batch_idx)
        enc_on._cluster_pool(agg, batch_idx)
        assert enc_off.aux_loss.item() < enc_on.aux_loss.item()

    def test_zero_weight_aux_still_positive_and_differentiable(self) -> None:
        """Node entropy remains as the aux loss and still carries gradient."""
        agg, batch_idx = _synthetic_nodes()
        agg.requires_grad_(True)
        enc = _make_encoder(0.0)
        enc._cluster_pool(agg, batch_idx)
        assert enc.aux_loss.item() > 0.0
        enc.aux_loss.backward()
        assert agg.grad is not None
        assert torch.isfinite(agg.grad).all()

    def test_usage_entropy_diagnostic_set_regardless_of_weight(self) -> None:
        """The detached usage-entropy diagnostic is recorded even when its loss weight is 0."""
        agg, batch_idx = _synthetic_nodes()
        for weight in (0.0, 1.0):
            enc = _make_encoder(weight)
            enc._cluster_pool(agg, batch_idx)
            assert enc.usage_entropy is not None
            assert not enc.usage_entropy.requires_grad
            assert 0.0 <= enc.usage_entropy.item() <= torch.log(torch.tensor(6.0)).item() + 1e-5
