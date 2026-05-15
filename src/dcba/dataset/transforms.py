"""Transforms for DCBA dataset records."""

from functools import lru_cache

import torch
from dcba_data_set.graph_io.data_models import DCBAHeteroData, DCBAInstanceConfig
from pydantic import BaseModel
from torch import Tensor
from torch_geometric.transforms import BaseTransform

#: Ordered list of numerical ABCD config keys used as model input features.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]


class ABCDConfigSchema(BaseModel):
    """Pydantic schema for the 9-parameter ABCD generator configuration.

    Field order matches :data:`ABCD_CONFIG_KEYS`.
    """

    n: float
    t1: float
    t2: float
    xi: float
    c_min: float
    c_max: float
    d_min: float
    d_max: float
    nout: float


#: Indices within :data:`ABCD_CONFIG_KEYS` that correspond to integer-valued parameters.
ABCD_INT_FEATURE_INDICES: list[int] = [0, 4, 5, 6, 7, 8]  # n, c_min, c_max, d_min, d_max, nout


@lru_cache()
def abcd_param_bounds(n_max: int = 10_000) -> dict[str, tuple[float, float]]:
    """
    Return per-feature ``(lo, hi)`` bounds for ABCD config normalisation.

    Bounds are derived from ABCDGraphGenerator.jl hard constraints (``src/pl_sampler.jl``,
    ``src/community_sampler.jl``, ``src/graph_sampler.jl``) and the canonical example configuration
    (``utils/example_config.toml``).

    ``n_max`` sets the upper bound for all features that are analytically bounded by ``n`` in the
    Julia model (``n``, ``c_min``, ``c_max``, ``d_min``, ``d_max``,``nout``):
        - ``t1``, ``t2``: ``@assert α >= 1`` (pl_sampler.jl); upper ~5 in practice.
        - ``xi``: ``0 ≤ ξ ≤ 1`` (hard constraint, graph_sampler.jl).
        - All others: lower bound from Julia assertions; upper bound ``= n_max``.

    :param n_max: Maximum graph size in the dataset.

    :returns: Dict mapping each key in :data:`ABCD_CONFIG_KEYS` to ``(lo, hi)``.
    """
    n = float(n_max)
    return {
        "n": (1.0, n),  # @assert n > 0; example: n = 10_000
        "t1": (1.0, 5.0),  # @assert α >= 1; power-law exponents rarely exceed 5
        "t2": (1.0, 5.0),  # @assert α >= 1; power-law exponents rarely exceed 5
        "xi": (0.0, 1.0),  # 0 ≤ ξ ≤ 1 (hard constraint)
        "c_min": (1.0, n),  # >= 1; c_min <= c_max <= n
        "c_max": (1.0, n),  # c_max <= n (ABCDConfig validator)
        "d_min": (1.0, n),  # @assert 1 <= d_min
        "d_max": (1.0, n),  # @assert d_max >= d_min
        "nout": (0.0, n),  # 0 <= nout <= n (ABCDConfig validator)
    }


class ABCDConfigScaler:
    """
    Normalise and denormalise ABCD config tensors feature-wise to ``[0, 1]``.

    Uses a linear (min-max) map per feature sourced from :func:`abcd_param_bounds`.

    The inverse is exact (linear map), making this usable at inference time to recover
    human-readable configs from model output.

    :param n_max: Maximum graph size in the dataset.  Passed directly to :func:`abcd_param_bounds`.
        Defaults to 10 000, matching the canonical ABCDGraphGenerator.jl example configuration.
    """

    def __init__(self, n_max: int = 10_000) -> None:
        """Initialise the scaler, building lo/hi tensors from :func:`abcd_param_bounds`."""
        bounds = abcd_param_bounds(n_max)
        self._lo = torch.tensor([bounds[k][0] for k in ABCD_CONFIG_KEYS], dtype=torch.float32)
        self._hi = torch.tensor([bounds[k][1] for k in ABCD_CONFIG_KEYS], dtype=torch.float32)

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        :param x: Float tensor of shape ``(..., 9)``.

        :returns: Normalised tensor of the same shape.
        """
        return (x - self._lo) / (self._hi - self._lo)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor.

        Integer-valued features (indices :data:`ABCD_INT_FEATURE_INDICES`) are rounded to the
        nearest whole number after the linear inverse map.  The tensor dtype remains ``float32``.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        out = x * (self._hi - self._lo) + self._lo
        out[..., ABCD_INT_FEATURE_INDICES] = out[..., ABCD_INT_FEATURE_INDICES].round()
        return out

    def __call__(self, x: Tensor) -> Tensor:
        """Apply :meth:`transform`."""
        return self.transform(x)


class CommunityToSize(BaseTransform):
    """
    Convert raw community IDs to normalised community-size node features.

    For each node and each layer, computes the fraction of *active* nodes
    (``community != 0``) that share the same community label.  Inactive nodes
    (``community == 0``, mABCD only) receive ``0.0``.

    Writes a float tensor of shape ``[num_actors, num_layers]`` into
    ``data["actor"].x``, replacing any existing value.
    """

    def forward(self, data: DCBAHeteroData) -> DCBAHeteroData:
        """
        Apply the transform to a single (non-batched) graph.

        :param data: A single heterogeneous graph.

        :returns: The same graph with ``data["actor"].x`` set to normalised community-size features.
        """
        community = data["actor"].community  # [N, L], long
        n, num_layers = community.shape
        sizes = torch.zeros(n, num_layers, dtype=torch.float32)
        for layer in range(num_layers):
            c = community[:, layer]
            active = c != 0
            n_active = int(active.sum())
            if n_active == 0:
                continue
            # inv maps each active node back to its index in the unique-values array,
            # so counts[inv] broadcasts the per-community count to a per-node count.
            _, inv, counts = torch.unique(c[active], return_inverse=True, return_counts=True)
            sizes[active, layer] = counts[inv].float() / n_active
        data["actor"].x = sizes
        return data


class ABCDConfigToTensor(BaseTransform):
    """
    Transform a :class:`~dcba_data_set.graph_io.data_models.DCBAInstanceConfig` to a float tensor.

    Extracts the 9 numerical ABCD config fields in the order defined by
    :data:`ABCD_CONFIG_KEYS` and returns a 1-D ``float32`` tensor of shape ``(9,)``.
    """

    def forward(self, data: DCBAInstanceConfig) -> Tensor:
        """
        Apply the transform.

        :param data: An ABCD config record.

        :returns: A 1-D float tensor of shape ``(9,)``.
        """
        values = [float(data.data[key]) for key in ABCD_CONFIG_KEYS]
        return torch.tensor(values, dtype=torch.float32)
