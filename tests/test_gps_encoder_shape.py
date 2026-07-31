"""Unit tests for GPSEncoderShape's size/density readout and modularity aux term."""

import math

import torch

from dcba.models.gps_encoder import GPSEncoder
from dcba.models.gps_encoder_shape import GPSEncoderShape

_HIDDEN_DIM = 8
_NUM_CLUSTERS = 6
_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
_READOUT_WIDTH = 2 * len(_QUANTILES) + 8


def _make_encoder(
    usage_entropy_weight: float = 0.0,
    modularity_weight: float = 0.0,
    node_entropy_weight: float = 1.0,
    num_clusters: int = _NUM_CLUSTERS,
) -> GPSEncoderShape:
    """Build a small encoder with the clustering branch enabled."""
    return GPSEncoderShape(
        hidden_dim=_HIDDEN_DIM,
        num_layers=1,
        num_heads=2,
        embedding_dim=_HIDDEN_DIM,
        output_dim=9,
        num_clusters=num_clusters,
        cluster_size_quantiles=_QUANTILES,
        usage_entropy_weight=usage_entropy_weight,
        node_entropy_weight=node_entropy_weight,
        modularity_weight=modularity_weight,
    )


def _synthetic_nodes(
    n_nodes: int = 30, hidden_dim: int = _HIDDEN_DIM
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return post-GPS node embeddings, a two-graph batch index, and a within-graph edge index."""
    torch.manual_seed(0)
    agg = torch.randn(n_nodes, hidden_dim)
    split = n_nodes // 2
    batch_idx = torch.cat([torch.zeros(split), torch.ones(n_nodes - split)]).long()
    # A ring per graph, so every node has degree 2 and no edge crosses the batch boundary.
    first = torch.arange(split)
    second = torch.arange(split, n_nodes)
    edges = torch.cat(
        [torch.stack([first, first.roll(-1)]), torch.stack([second, second.roll(-1)])], dim=-1
    )
    edge_index = torch.cat([edges, edges.flip(0)], dim=-1)
    return agg, batch_idx, edge_index


def _slots(
    size: list[float], density: list[float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ``(size, intra_volume, volume)`` for one graph from per-slot sizes and densities."""
    size_t = torch.tensor([size])
    volume = size_t * 4.0  # an arbitrary but positive degree per node
    return size_t, volume * torch.tensor([density]), volume


class TestAuxEntropyTerms:
    """The entropy terms inherited from GPSEncoder must keep behaving the same way."""

    def test_zero_weight_drops_usage_term(self) -> None:
        """With weight 0 the aux loss is strictly smaller than with weight 1 on the same input."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        enc_off, enc_on = _make_encoder(0.0), _make_encoder(1.0)
        enc_on.load_state_dict(enc_off.state_dict())  # identical parameters, same assignments

        enc_off._shape_pool(agg, batch_idx, edge_index)
        enc_on._shape_pool(agg, batch_idx, edge_index)
        assert enc_off.aux_loss.item() < enc_on.aux_loss.item()

    def test_zero_weight_aux_still_positive_and_differentiable(self) -> None:
        """Node entropy remains as the aux loss and still carries gradient."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        agg.requires_grad_(True)
        enc = _make_encoder()
        enc._shape_pool(agg, batch_idx, edge_index)
        assert enc.aux_loss.item() > 0.0
        enc.aux_loss.backward()
        assert agg.grad is not None
        assert torch.isfinite(agg.grad).all()

    def test_usage_entropy_diagnostic_set_regardless_of_weight(self) -> None:
        """The detached usage-entropy diagnostic is recorded even when its loss weight is 0."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        for weight in (0.0, 1.0):
            enc = _make_encoder(weight)
            enc._shape_pool(agg, batch_idx, edge_index)
            assert enc.usage_entropy is not None
            assert not enc.usage_entropy.requires_grad
            assert 0.0 <= enc.usage_entropy.item() <= math.log(_NUM_CLUSTERS) + 1e-5


class TestSizeProfile:
    """The size half of the readout must be scale-free and invariant to unused slots."""

    def test_readout_width(self) -> None:
        """Two Q-wide profiles plus the four density extremes and four scalar channels."""
        enc = _make_encoder()
        readout = enc._readout(*_slots([6, 3, 1, 0, 0, 0], [0.9, 0.8, 0.7, 0, 0, 0]))
        assert readout.shape == (1, _READOUT_WIDTH)

    def test_invariant_to_graph_size(self) -> None:
        """Scaling every cluster by the same factor leaves the size profile unchanged."""
        enc = _make_encoder()
        size, intra, volume = _slots([60, 30, 10, 0, 0, 0], [0.9, 0.8, 0.7, 0, 0, 0])
        profile = enc._readout(size, intra, volume)[:, : len(_QUANTILES)]
        scaled = enc._readout(size * 100, intra * 100, volume * 100)[:, : len(_QUANTILES)]
        assert torch.allclose(profile, scaled)

    def test_invariant_to_unused_slots(self) -> None:
        """Padding the budget with empty slots must not move the size profile."""
        enc = _make_encoder()
        occupied = enc._readout(*_slots([60, 30, 10], [0.9, 0.8, 0.7]))
        padded = enc._readout(*_slots([60, 30, 10, 0, 0, 0], [0.9, 0.8, 0.7, 0, 0, 0]))
        assert torch.allclose(
            occupied[:, : len(_QUANTILES)], padded[:, : len(_QUANTILES)], atol=1e-6
        )

    def test_bounded_channels(self) -> None:
        """Every channel stays in [0, 1] for a plausible slot configuration."""
        enc = _make_encoder()
        readout = enc._readout(
            *_slots([500, 250, 120, 80, 40, 10], [0.9, 0.85, 0.8, 0.7, 0.5, 0.05])
        )
        assert readout.min() >= 0.0
        assert readout.max() <= 1.0 + 1e-6

    def test_occupancy_channel_tracks_used_slots(self) -> None:
        """The occupancy channel is near 1/K for one slot and 1 when all slots are equal."""
        enc = _make_encoder()
        one_slot = enc._readout(*_slots([30, 0, 0, 0, 0, 0], [0.9, 0, 0, 0, 0, 0]))
        uniform = enc._readout(*_slots([5] * 6, [0.9] * 6))
        assert one_slot[0, -2].item() < 1.5 / _NUM_CLUSTERS
        assert uniform[0, -2].item() > 0.99

    def test_readout_is_differentiable(self) -> None:
        """Gradients flow through the gathered order statistics."""
        enc = _make_encoder()
        size, intra, volume = _slots([6, 3, 1, 0.5, 0, 0], [0.9, 0.8, 0.7, 0.1, 0, 0])
        size.requires_grad_(True)
        intra.requires_grad_(True)
        enc._readout(size, intra, volume).sum().backward()
        for tensor in (size, intra):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()
            assert tensor.grad.abs().sum() > 0.0


class TestDensityAndCutChannels:
    """The density profile and cut channel are what separate this readout from GPSEncoder's."""

    def test_density_profile_follows_slot_density(self) -> None:
        """Read at the size-biased positions, the profile reports each slot's own density."""
        enc = _make_encoder()
        readout = enc._readout(*_slots([100, 1, 1, 1, 1, 1], [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]))
        density = readout[0, len(_QUANTILES) : 2 * len(_QUANTILES)]
        # Every quantile level up to 0.75 lands inside the dominant slot for this configuration.
        assert torch.allclose(density[:4], torch.full((4,), 0.9), atol=1e-5)

    def test_density_separates_outlier_set_from_small_community(self) -> None:
        """Same sizes, different densities -- a size-only readout could not tell these apart."""
        enc = _make_encoder()
        community = enc._readout(*_slots([100, 20], [0.9, 0.85]))
        outliers = enc._readout(*_slots([100, 20], [0.9, 0.02]))
        size_half = slice(0, len(_QUANTILES))
        density_half = slice(len(_QUANTILES), 2 * len(_QUANTILES))
        assert torch.allclose(community[:, size_half], outliers[:, size_half])
        assert not torch.allclose(community[:, density_half], outliers[:, density_half])

    def test_density_extremes_ignore_empty_slots(self) -> None:
        """Regression: an empty slot has zero density and would otherwise win every argmin.

        With ``num_clusters`` set above the true community count most slots are empty, so an
        unmasked minimum would report 0.0 for every graph and the channel would be constant.
        """
        enc = _make_encoder()
        extremes = slice(2 * len(_QUANTILES), 2 * len(_QUANTILES) + 4)
        occupied = enc._readout(*_slots([60, 30, 10], [0.9, 0.8, 0.55]))[:, extremes]
        padded = enc._readout(*_slots([60, 30, 10, 0, 0, 0], [0.9, 0.8, 0.55, 0, 0, 0]))[
            :, extremes
        ]
        assert torch.allclose(occupied, padded, atol=1e-6)
        assert math.isclose(padded[0, 0].item(), 0.55, rel_tol=1e-5)  # least dense OCCUPIED slot
        assert math.isclose(padded[0, 1].item(), 0.9, rel_tol=1e-5)  # most dense slot

    def test_density_extremes_locate_the_outlier_slot(self) -> None:
        """A large sparse slot alongside dense communities is reported by mass, not just density."""
        enc = _make_encoder()
        extremes = slice(2 * len(_QUANTILES), 2 * len(_QUANTILES) + 4)
        # 40 nodes keeping almost nothing internally, against 160 in three dense communities.
        readout = enc._readout(*_slots([80, 50, 30, 40], [0.9, 0.88, 0.85, 0.02]))[:, extremes]
        assert math.isclose(readout[0, 0].item(), 0.02, rel_tol=1e-5)
        assert math.isclose(readout[0, 2].item(), 40 / 200, rel_tol=1e-5)  # its share of nodes
        assert math.isclose(readout[0, 3].item(), 40 / 80, rel_tol=1e-5)  # its size vs the largest

    def test_cut_channel_is_share_of_volume_kept_inside(self) -> None:
        """The final channel is total intra volume over total volume."""
        enc = _make_encoder()
        size, intra, volume = _slots([10, 10], [0.8, 0.4])
        readout = enc._readout(size, intra, volume)
        assert math.isclose(readout[0, -1].item(), 0.6, rel_tol=1e-5)

    def test_cut_diagnostic_always_recorded(self) -> None:
        """The cut is a readout channel, so it is computed regardless of any loss weight."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        enc = _make_encoder(modularity_weight=0.0)
        enc._shape_pool(agg, batch_idx, edge_index)
        assert enc.normalised_cut is not None
        assert not enc.normalised_cut.requires_grad
        assert 0.0 <= enc.normalised_cut.item() <= 1.0 + 1e-6


class TestModularityTerm:
    """Modularity ties the slots to the edges; it must have no degenerate optimum."""

    def test_collapse_scores_zero(self) -> None:
        """One slot holding the whole graph scores exactly 0 -- collapse is not an optimum.

        This is the property the MinCutPool normalised cut lacks: the same partition maximises the
        cut at 1.0, which is why that objective needs an orthogonality counterweight.
        """
        volume = torch.tensor([[40.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        intra = volume.clone()  # every edge internal to the single occupied slot
        assert abs(GPSEncoderShape._modularity(intra, volume).item()) < 1e-6

    def test_true_partition_beats_fragmenting_it(self) -> None:
        """Splitting a dense group forfeits internal edges, so modularity drops."""
        whole = GPSEncoderShape._modularity(
            torch.tensor([[36.0, 36.0]]), torch.tensor([[40.0, 40.0]])
        )
        # Same volume split across four slots, each keeping only half its edges internally.
        split = GPSEncoderShape._modularity(
            torch.tensor([[9.0, 9.0, 9.0, 9.0]]), torch.tensor([[20.0, 20.0, 20.0, 20.0]])
        )
        assert whole.item() > split.item()

    def test_random_assignment_scores_near_zero(self) -> None:
        """A partition whose intra volume matches the null model scores ~0."""
        volume = torch.tensor([[25.0, 25.0, 25.0, 25.0]])
        intra = volume * 0.25  # exactly the null-model expectation for four equal slots
        assert abs(GPSEncoderShape._modularity(intra, volume).item()) < 1e-6

    def test_bounded_above_by_one(self) -> None:
        """A perfect many-community partition approaches 1 without exceeding it."""
        volume = torch.full((1, 6), 20.0)
        assert GPSEncoderShape._modularity(volume, volume).item() < 1.0

    def test_diagnostic_recorded_without_weight(self) -> None:
        """Modularity is logged whether or not it is weighted into the loss."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        enc = _make_encoder(modularity_weight=0.0)
        enc._shape_pool(agg, batch_idx, edge_index)
        assert enc.modularity is not None
        assert not enc.modularity.requires_grad

    def test_weight_actually_changes_the_gradient(self) -> None:
        """Regression: the term must reach aux_loss undetached, or it is a silent no-op."""
        parameters = _make_encoder().state_dict()  # identical in both runs, only the weight differs
        gradients = []
        for weight in (0.0, 1.0):
            agg, batch_idx, edge_index = _synthetic_nodes()
            agg.requires_grad_(True)
            enc = _make_encoder(modularity_weight=weight)
            enc.load_state_dict(parameters)
            enc._shape_pool(agg, batch_idx, edge_index)
            enc.aux_loss.backward()
            gradients.append(agg.grad.clone())
        assert not torch.allclose(gradients[0], gradients[1])

    def test_gradients_finite_when_enabled(self) -> None:
        """Backward through the sparse per-slot statistics produces finite gradients."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        agg.requires_grad_(True)
        enc = _make_encoder(modularity_weight=1.0)
        enc._shape_pool(agg, batch_idx, edge_index)
        enc.aux_loss.backward()
        assert agg.grad is not None
        assert torch.isfinite(agg.grad).all()


class TestInheritance:
    """Only the readout may differ from GPSEncoder; everything else is inherited."""

    def test_is_a_gps_encoder(self) -> None:
        """It subclasses GPSEncoder, so the GPS stack and decoder are shared code."""
        assert issubclass(GPSEncoderShape, GPSEncoder)

    def test_projection_widened_by_the_extra_channels(self) -> None:
        """_proj grows by exactly the density profile plus the four scalars."""
        shape = _make_encoder()
        legacy = GPSEncoder(
            hidden_dim=_HIDDEN_DIM,
            num_layers=1,
            num_heads=2,
            embedding_dim=_HIDDEN_DIM,
            output_dim=9,
            num_clusters=_NUM_CLUSTERS,
            cluster_size_quantiles=_QUANTILES,
        )
        expected = len(_QUANTILES) + 8
        assert shape._proj[0].in_features - legacy._proj[0].in_features == expected

    def test_projection_untouched_without_the_branch(self) -> None:
        """With num_clusters 0 the branch is off and _proj matches the parent exactly."""
        assert _make_encoder(num_clusters=0)._proj[0].in_features == _HIDDEN_DIM

    def test_decode_and_forward_are_inherited(self) -> None:
        """Decode is not overridden, so the inherited decoder shapes still hold."""
        assert GPSEncoderShape.decode is GPSEncoder.decode
        assert GPSEncoderShape.forward is GPSEncoder.forward
        enc = _make_encoder()
        assert enc.decode(torch.randn(3, _HIDDEN_DIM)).shape == (3, 9)


class TestNodeEntropyWeight:
    """The inherited per-node entropy term must be switchable, since it can outvote modularity."""

    def test_default_reproduces_the_inherited_term(self) -> None:
        """At weight 1.0 the term contributes its full value, as GPSEncoder does."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        full, off = _make_encoder(node_entropy_weight=1.0), _make_encoder(node_entropy_weight=0.0)
        off.load_state_dict(full.state_dict())
        full._shape_pool(agg, batch_idx, edge_index)
        off._shape_pool(agg, batch_idx, edge_index)
        assert full.aux_loss.item() > off.aux_loss.item()

    def test_zero_weight_leaves_only_modularity(self) -> None:
        """With both entropies off, aux_loss is exactly the negated weighted modularity."""
        agg, batch_idx, edge_index = _synthetic_nodes()
        enc = _make_encoder(node_entropy_weight=0.0, modularity_weight=1.0)
        enc._shape_pool(agg, batch_idx, edge_index)
        assert math.isclose(enc.aux_loss.item(), -enc.modularity.item(), rel_tol=1e-5)

    def test_zero_weight_removes_the_collapse_pressure(self) -> None:
        """The term's gradient dominates modularity's, which is what collapsed run z7pulj4r.

        log(num_clusters) is the term's range and dwarfs modularity's, so with it on the aux loss is
        driven by sharpening rather than by community structure.
        """
        agg, batch_idx, edge_index = _synthetic_nodes()
        enc = _make_encoder(node_entropy_weight=1.0, modularity_weight=1.0)
        enc._shape_pool(agg, batch_idx, edge_index)
        # At initialisation the assignment is near-uniform, so node entropy sits close to log(K)
        # while modularity is near 0 -- the imbalance the config's 0.0 exists to remove.
        assert enc.aux_loss.item() > 0.5 * math.log(_NUM_CLUSTERS)
        assert abs(enc.modularity.item()) < 0.1
