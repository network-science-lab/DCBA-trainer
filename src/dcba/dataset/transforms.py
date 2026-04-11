"""Transforms for DCBA dataset records."""

from functools import lru_cache

import torch
from dcba_data_set.graph_io.data_models import ConfigRecord
from torch import Tensor
from torch_geometric.transforms import BaseTransform

#: Ordered list of numerical ABCD config keys used as model input features.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]


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

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        return x * (self._hi - self._lo) + self._lo

    def __call__(self, x: Tensor) -> Tensor:
        """Apply :meth:`transform`."""
        return self.transform(x)


class ABCDConfigToTensor(BaseTransform):
    """
    Transform an ABCD :class:`~dcba_data_set.graph_io.data_models.ConfigRecord` to a float tensor.

    Extracts the 9 numerical ABCD config fields in the order defined by
    :data:`ABCD_CONFIG_KEYS` and returns a 1-D ``float32`` tensor of shape ``(9,)``.
    """

    def forward(self, data: ConfigRecord) -> Tensor:
        """
        Apply the transform.

        :param data: An ABCD config record.

        :returns: A 1-D float tensor of shape ``(9,)``.
        """
        values = [float(data.data[key]) for key in ABCD_CONFIG_KEYS]
        return torch.tensor(values, dtype=torch.float32)
