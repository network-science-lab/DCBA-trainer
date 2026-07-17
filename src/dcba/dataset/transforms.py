"""Transforms for DCBA dataset records."""

import math
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

    Bounds combine ABCDGraphGenerator.jl hard constraints (``t1``, ``t2`` >= 1; ``xi`` in
    ``[0, 1]``) with per-feature maxima observed on ``abcd-big``. Each size-like feature gets its
    own bound rather than a shared ``n_max``, so none is starved into a sliver of ``[0, 1]``.
    ``c_min`` is bounded ``[0, 1]`` because :class:`ABCDConfigScaler` scales it as a fraction of
    ``c_max``, not as an absolute count.

    :param n_max: Maximum graph size in the dataset; used only for the ``n`` bound.

    :returns: Dict mapping each key in :data:`ABCD_CONFIG_KEYS` to ``(lo, hi)``.
    """
    n = float(n_max)
    return {
        "n": (1.0, n),  # @assert n > 0
        "t1": (1.0, 5.0),  # @assert α >= 1
        "t2": (1.0, 5.0),  # @assert α >= 1
        "xi": (0.0, 1.0),  # 0 ≤ ξ ≤ 1 (hard constraint)
        "c_min": (0.0, 1.0),  # fraction of c_max -- see ABCDConfigScaler
        "c_max": (1.0, 6_000.0),  # observed max ~4938 on abcd-big
        "d_min": (1.0, 100.0),  # observed max ~50 on abcd-big
        "d_max": (1.0, 6_000.0),  # observed max ~4993 on abcd-big
        "nout": (0.0, 700.0),  # observed max ~492 on abcd-big
    }


class ABCDConfigScaler:
    """
    Normalise and denormalise ABCD config tensors feature-wise to ``[0, 1]``.

    Uses a linear (min-max) map per feature sourced from :func:`abcd_param_bounds`, except
    ``c_min``, which is scaled as ``c_min / c_max`` instead (both already ``[0, 1]``-bounded by
    ``abcd_param_bounds`` -- see its docstring for why a fixed bound does not work for ``c_min``).

    The inverse is exact, making this usable at inference time to recover human-readable configs
    from model output.

    :param n_max: Maximum graph size in the dataset.  Passed directly to :func:`abcd_param_bounds`.
        Defaults to 10 000, matching the canonical ABCDGraphGenerator.jl example configuration.
    """

    def __init__(self, n_max: int = 10_000) -> None:
        """Initialise the scaler, building lo/hi tensors from :func:`abcd_param_bounds`."""
        bounds = abcd_param_bounds(n_max)
        self._lo = torch.tensor([bounds[k][0] for k in ABCD_CONFIG_KEYS], dtype=torch.float32)
        self._hi = torch.tensor([bounds[k][1] for k in ABCD_CONFIG_KEYS], dtype=torch.float32)
        self._c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        self._c_max_idx = ABCD_CONFIG_KEYS.index("c_max")

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = (x - self._lo) / (self._hi - self._lo)
        c_max_raw = x[..., self._c_max_idx].clamp(min=1.0)
        out[..., self._c_min_idx] = x[..., self._c_min_idx] / c_max_raw
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable -- built column-wise without in-place writes so gradients can flow
        through the inverse map (e.g. raw-space constraint penalties in
        :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss`). Use
        :meth:`inverse_transform` instead when integer-rounded configs are wanted.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale, integer features left unrounded.
        """
        lo = self._lo.to(x.device)
        hi = self._hi.to(x.device)
        lin = x * (hi - lo) + lo
        cols = list(lin.unbind(dim=-1))
        # c_max must already be in its final (raw) scale before c_min can be recovered from it.
        cols[self._c_min_idx] = x[..., self._c_min_idx] * cols[self._c_max_idx]
        return torch.stack(cols, dim=-1)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor.

        Integer-valued features (indices :data:`ABCD_INT_FEATURE_INDICES`) are rounded to the
        nearest whole number after the linear inverse map.  The tensor dtype remains ``float32``.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        out = self.denormalise(x)
        out[..., ABCD_INT_FEATURE_INDICES] = out[..., ABCD_INT_FEATURE_INDICES].round()
        return out

    def __call__(self, x: Tensor) -> Tensor:
        """Apply :meth:`transform`."""
        return self.transform(x)


#: Size-like features scaled as ``log1p(x) / log1p(n)`` by :class:`ABCDLogConfigScaler`.
_LOG_N_RELATIVE_KEYS: tuple[str, ...] = ("c_max", "d_min", "d_max", "nout")


class ABCDLogConfigScaler:
    """
    Normalise ABCD config tensors to ``[0, 1]``, log-compressing size-like features against `n`.

    Maps ``c_max``, ``d_min``, ``d_max`` and ``nout`` as ``log1p(x) / log1p(n)`` -- relative to
    each graph's own `n` rather than a fixed bound, so the map adapts to any `n` and spreads
    heavy-tailed values across ``[0, 1]``. `n` itself is scaled as ``log1p(n) / log1p(n_max)``.
    `t1`, `t2`, `xi` keep :class:`ABCDConfigScaler`'s linear map and `c_min` its ``c_min / c_max``
    ratio.

    The inverse is exact: :meth:`inverse_transform` denormalises `n` first (every log-relative
    feature depends on it), then `c_max` before `c_min`.

    :param n_max: Maximum graph size in the dataset, used only to scale `n`. Defaults to 10 000.
    """

    def __init__(self, n_max: int = 10_000) -> None:
        """Initialise the scaler, caching feature indices and the linear bounds it still uses."""
        self._n_max = float(n_max)
        self._log_n_max = math.log1p(self._n_max)
        bounds = abcd_param_bounds(n_max)
        linear_keys = ("t1", "t2", "xi")
        self._linear_indices = [ABCD_CONFIG_KEYS.index(k) for k in linear_keys]
        self._linear_lo = torch.tensor([bounds[k][0] for k in linear_keys], dtype=torch.float32)
        self._linear_hi = torch.tensor([bounds[k][1] for k in linear_keys], dtype=torch.float32)
        self._n_idx = ABCD_CONFIG_KEYS.index("n")
        self._c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        self._c_max_idx = ABCD_CONFIG_KEYS.index("c_max")
        self._log_indices = [ABCD_CONFIG_KEYS.index(k) for k in _LOG_N_RELATIVE_KEYS]

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature, log-compressing size-like features against `n`.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = torch.zeros_like(x)
        n_raw = x[..., self._n_idx].clamp(min=1.0)
        log_n = torch.log1p(n_raw)

        out[..., self._n_idx] = log_n / self._log_n_max
        out[..., self._linear_indices] = (x[..., self._linear_indices] - self._linear_lo) / (
            self._linear_hi - self._linear_lo
        )
        out[..., self._log_indices] = torch.log1p(
            x[..., self._log_indices].clamp(min=0.0)
        ) / log_n.unsqueeze(-1)

        c_max_raw = x[..., self._c_max_idx].clamp(min=1.0)
        out[..., self._c_min_idx] = x[..., self._c_min_idx] / c_max_raw
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable counterpart of :meth:`inverse_transform`, built column-wise without
        in-place writes -- see :meth:`ABCDConfigScaler.denormalise` for when to use which.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale, integer features left unrounded.
        """
        lo = self._linear_lo.to(x.device)
        hi = self._linear_hi.to(x.device)
        n_raw = torch.expm1(x[..., self._n_idx] * self._log_n_max).clamp(min=1.0)
        log_n = torch.log1p(n_raw)
        lin = x[..., self._linear_indices] * (hi - lo) + lo
        logs = torch.expm1(x[..., self._log_indices] * log_n.unsqueeze(-1))

        cols: list[Tensor] = [torch.empty(0)] * len(ABCD_CONFIG_KEYS)
        cols[self._n_idx] = n_raw
        for j, i in enumerate(self._linear_indices):
            cols[i] = lin[..., j]
        for j, i in enumerate(self._log_indices):
            cols[i] = logs[..., j]
        cols[self._c_min_idx] = x[..., self._c_min_idx] * cols[self._c_max_idx].clamp(min=1.0)
        return torch.stack(cols, dim=-1)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor.

        `n` is denormalised first since every log-relative feature's inverse depends on it, and
        `c_max` (also log-relative) before `c_min`, which is recovered as a fraction of it.
        Integer-valued features (:data:`ABCD_INT_FEATURE_INDICES`) are rounded to the nearest
        whole number after the inverse map. The tensor dtype remains ``float32``.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        out = self.denormalise(x)
        out[..., ABCD_INT_FEATURE_INDICES] = out[..., ABCD_INT_FEATURE_INDICES].round()
        return out

    def __call__(self, x: Tensor) -> Tensor:
        """Apply :meth:`transform`."""
        return self.transform(x)


class ABCDRelativeConfigScaler:
    """
    Normalise ABCD config tensors to ``[0, 1]`` using hard-constraint anchors readable from any
    graph, with no dataset-estimated bounds on the size-like features.

    - ``c_max / n`` and ``d_max / n`` (hard constraint ``x <= n``).
    - ``d_min -> log(d_min) / log(d_max)`` (hard constraint ``1 <= d_min <= d_max``), i.e. the
      exponent ``s`` such that ``d_min = d_max ** s``.
    - ``nout -> log1p(nout) / log1p(n)`` (hard constraint ``0 <= nout <= n``); ``log1p`` handles
      ``nout = 0``.
    - `n`, `t1`, `t2`, `xi` keep :class:`ABCDConfigScaler`'s linear map and `c_min` its
      ``c_min / c_max`` ratio, on which :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss`
      relies for the ``c_min <= c_max`` guarantee.

    The inverse is exact: `n` is denormalised first, then `d_max` before `d_min` and `c_max`
    before `c_min`.

    :param n_max: Maximum graph size in the dataset, used only for `n`. Defaults to 10 000.
    """

    def __init__(self, n_max: int = 10_000) -> None:
        """Initialise the scaler, caching feature indices and the linear bounds it still uses."""
        bounds = abcd_param_bounds(n_max)
        linear_keys = ("t1", "t2", "xi")
        self._linear_indices = [ABCD_CONFIG_KEYS.index(k) for k in linear_keys]
        self._linear_lo = torch.tensor([bounds[k][0] for k in linear_keys], dtype=torch.float32)
        self._linear_hi = torch.tensor([bounds[k][1] for k in linear_keys], dtype=torch.float32)
        self._n_idx = ABCD_CONFIG_KEYS.index("n")
        self._n_lo, self._n_hi = bounds["n"]
        self._c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        self._c_max_idx = ABCD_CONFIG_KEYS.index("c_max")
        self._d_min_idx = ABCD_CONFIG_KEYS.index("d_min")
        self._d_max_idx = ABCD_CONFIG_KEYS.index("d_max")
        self._nout_idx = ABCD_CONFIG_KEYS.index("nout")

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature, scaling size-like features against their
        hard-constraint anchors (`n`, `d_max`).

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = torch.zeros_like(x)
        n_raw = x[..., self._n_idx].clamp(min=1.0)

        out[..., self._n_idx] = (n_raw - self._n_lo) / (self._n_hi - self._n_lo)
        out[..., self._linear_indices] = (x[..., self._linear_indices] - self._linear_lo) / (
            self._linear_hi - self._linear_lo
        )

        out[..., self._c_max_idx] = x[..., self._c_max_idx] / n_raw
        out[..., self._d_max_idx] = x[..., self._d_max_idx] / n_raw
        # d_max clamped to >= 2 so log(d_max) > 0; only reachable on degenerate raw configs.
        out[..., self._d_min_idx] = torch.log(x[..., self._d_min_idx].clamp(min=1.0)) / torch.log(
            x[..., self._d_max_idx].clamp(min=2.0)
        )
        out[..., self._nout_idx] = torch.log1p(x[..., self._nout_idx].clamp(min=0.0)) / torch.log1p(
            n_raw
        )

        c_max_raw = x[..., self._c_max_idx].clamp(min=1.0)
        out[..., self._c_min_idx] = x[..., self._c_min_idx] / c_max_raw
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable counterpart of :meth:`inverse_transform`, built column-wise without
        in-place writes -- see :meth:`ABCDConfigScaler.denormalise` for when to use which.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale, integer features left unrounded.
        """
        lo = self._linear_lo.to(x.device)
        hi = self._linear_hi.to(x.device)
        n_raw = (x[..., self._n_idx] * (self._n_hi - self._n_lo) + self._n_lo).clamp(min=1.0)
        lin = x[..., self._linear_indices] * (hi - lo) + lo

        cols: list[Tensor] = [torch.empty(0)] * len(ABCD_CONFIG_KEYS)
        cols[self._n_idx] = n_raw
        for j, i in enumerate(self._linear_indices):
            cols[i] = lin[..., j]

        c_max_raw = x[..., self._c_max_idx] * n_raw
        d_max_raw = x[..., self._d_max_idx] * n_raw
        cols[self._c_max_idx] = c_max_raw
        cols[self._d_max_idx] = d_max_raw
        cols[self._d_min_idx] = d_max_raw.clamp(min=2.0) ** x[..., self._d_min_idx]
        cols[self._nout_idx] = torch.expm1(x[..., self._nout_idx] * torch.log1p(n_raw))
        cols[self._c_min_idx] = x[..., self._c_min_idx] * c_max_raw.clamp(min=1.0)
        return torch.stack(cols, dim=-1)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor.

        `n` is denormalised first since every relative feature's inverse depends on it, then
        `d_max` before `d_min` (recovered as ``d_max ** s``) and `c_max` before `c_min`
        (recovered as a fraction of it). Integer-valued features
        (:data:`ABCD_INT_FEATURE_INDICES`) are rounded to the nearest whole number after the
        inverse map. The tensor dtype remains ``float32``.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        out = self.denormalise(x)
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


class ConstantNodeFeatures(BaseTransform):
    """
    Set node features to a constant, carrying no community information at all.

    Ablation baseline for :class:`CommunityToSize`, to isolate how much of the graph encoder's
    predictive power actually comes from the community-size feature versus pure graph structure.
    Writes a tensor of ones with the same shape ``[num_actors, num_layers]`` that
    :class:`CommunityToSize` would produce, so :class:`~dcba.models.gps_encoder.GPSEncoder` needs
    no changes to consume it.
    """

    def forward(self, data: DCBAHeteroData) -> DCBAHeteroData:
        """
        Apply the transform to a single (non-batched) graph.

        :param data: A single heterogeneous graph.

        :returns: The same graph with ``data["actor"].x`` set to all ones.
        """
        n, num_layers = data["actor"].community.shape
        data["actor"].x = torch.ones(n, num_layers, dtype=torch.float32)
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
