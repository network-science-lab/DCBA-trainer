"""Scalers and theta-vector transforms for ABCD config records."""

import math

import torch
from dcba_data_set.graph_io.data_models import DCBAInstanceConfig
from pydantic import BaseModel
from torch import Tensor
from torch_geometric.transforms import BaseTransform

#: Ordered list of numerical ABCD config keys used as model input features.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]

ABCD_INT_FEATURE_INDICES: list[int] = [0, 4, 5, 6, 7, 8]

_LOG_N_RELATIVE_KEYS: tuple[str, ...] = ("c_max", "d_min", "d_max", "nout")


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


#: Per-feature ``(lo, hi)`` bounds for ABCD config normalisation. ``t1``/``t2`` >= 1 and ``xi``
#: in ``[0, 1]`` are hard generator constraints; the rest -- including ``t1``/``t2``'s upper
#: bound -- are chosen ceilings, several observed on ``abcd-big``. ``c_min``'s bound is
#: ``[0, 1]`` since :class:`ABCDConfigScaler` scales it as a fraction of ``c_max``.
ABCD_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "n": (1.0, 10_000.0),  # @assert n > 0; upper bound is a chosen ceiling
    "t1": (1.0, 5.0),  # @assert alpha >= 1; upper bound is a chosen ceiling
    "t2": (1.0, 5.0),  # @assert alpha >= 1; upper bound is a chosen ceiling
    "xi": (0.0, 1.0),  # 0 <= xi <= 1 (hard constraint)
    "c_min": (0.0, 1.0),  # fraction of c_max -- see ABCDConfigScaler
    "c_max": (1.0, 6_000.0),  # observed max ~4938 on abcd-big
    "d_min": (1.0, 100.0),  # observed max ~50 on abcd-big
    "d_max": (1.0, 6_000.0),  # observed max ~4993 on abcd-big
    "nout": (0.0, 700.0),  # observed max ~492 on abcd-big
}

#: Per-feature ``(lo, hi)`` bounds for :class:`ABCDNMaxConfigScaler`: every size feature shares
#: a ``10_000`` upper bound (a chosen ceiling); ``t1``/``t2``/``xi`` keep their own bounds.
ABCD_NMAX_BOUNDS: dict[str, tuple[float, float]] = {
    "n": (1.0, 10_000.0),  # @assert n > 0
    "t1": (1.0, 5.0),  # @assert alpha >= 1
    "t2": (1.0, 5.0),  # @assert alpha >= 1
    "xi": (0.0, 1.0),  # 0 <= xi <= 1 (hard constraint)
    "c_min": (1.0, 10_000.0),  # >= 1; c_min <= c_max <= n
    "c_max": (1.0, 10_000.0),  # c_max <= n (ABCDConfig validator)
    "d_min": (1.0, 10_000.0),  # @assert 1 <= d_min
    "d_max": (1.0, 10_000.0),  # @assert d_max >= d_min
    "nout": (0.0, 10_000.0),  # 0 <= nout <= n (ABCDConfig validator)
}


class ABCDBaseConfigScaler:
    """
    Shared scaffolding for the ``[0, 1]`` ABCD config scalers below.

    Provides the ``c_min/c_max`` ratio helpers (unused by :class:`ABCDNMaxConfigScaler`), the
    integer-rounding :meth:`inverse_transform`, and :meth:`__call__`. Subclasses implement
    :meth:`transform`/:meth:`denormalise`.
    """

    def __init__(self) -> None:
        """Cache the ``c_min``/``c_max`` feature indices used by the ratio helpers below."""
        self._c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        self._c_max_idx = ABCD_CONFIG_KEYS.index("c_max")

    def transform(self, x: Tensor) -> Tensor:
        """Normalise ``x`` to ``[0, 1]`` per feature. Implemented by each subclass."""
        raise NotImplementedError

    def denormalise(self, x: Tensor) -> Tensor:
        """Recover the original scale from a normalised tensor. Implemented by each subclass."""
        raise NotImplementedError

    def inverse_transform(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor.

        Integer-valued features (indices :data:`ABCD_INT_FEATURE_INDICES`) are rounded to the
        nearest whole number after :meth:`denormalise`. The tensor dtype remains ``float32``.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale.
        """
        out = self.denormalise(x)
        out[..., ABCD_INT_FEATURE_INDICES] = out[..., ABCD_INT_FEATURE_INDICES].round()
        return out

    def __call__(self, x: Tensor) -> Tensor:
        """Apply :meth:`transform`."""
        return self.transform(x)

    def _scale_c_min(self, x: Tensor) -> Tensor:
        """
        Compute ``c_min`` as a fraction of raw ``c_max`` (clamped >= 1 to avoid a div by zero).

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised ``c_min`` column, shape ``(...,)``.
        """
        c_max_raw = x[..., self._c_max_idx].clamp(min=1.0)
        return x[..., self._c_min_idx] / c_max_raw

    def _unscale_c_min(self, x: Tensor, c_max_raw: Tensor) -> Tensor:
        """
        Recover raw ``c_min`` from its ``c_min / c_max`` ratio.

        :param x: Normalised float tensor of shape ``(..., 9)`` (only its ``c_min`` column is
            read).
        :param c_max_raw: Already-recovered raw ``c_max``. Clamp before calling if needed; not
            applied here.

        :returns: Raw ``c_min`` column, shape ``(...,)``.
        """
        return x[..., self._c_min_idx] * c_max_raw


class ABCDConfigScaler(ABCDBaseConfigScaler):
    """
    Normalise and denormalise ABCD config tensors feature-wise to ``[0, 1]``.

    Linear (min-max) map per feature from :data:`ABCD_PARAM_BOUNDS`, except ``c_min``, scaled as
    ``c_min / c_max``. The inverse is exact.
    """

    def __init__(self) -> None:
        """Initialise the scaler, building lo/hi tensors from :data:`ABCD_PARAM_BOUNDS`."""
        super().__init__()
        self._lo = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][0] for k in ABCD_CONFIG_KEYS], dtype=torch.float32
        )
        self._hi = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][1] for k in ABCD_CONFIG_KEYS], dtype=torch.float32
        )

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = (x - self._lo) / (self._hi - self._lo)
        out[..., self._c_min_idx] = self._scale_c_min(x)
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable. Use :meth:`inverse_transform` for integer-rounded output.

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale, integer features left unrounded.
        """
        lo = self._lo.to(x.device)
        hi = self._hi.to(x.device)
        lin = x * (hi - lo) + lo
        cols = list(lin.unbind(dim=-1))
        # c_max must already be in its final (raw) scale before c_min can be recovered from it.
        cols[self._c_min_idx] = self._unscale_c_min(x, cols[self._c_max_idx])
        return torch.stack(cols, dim=-1)


class ABCDNMaxConfigScaler(ABCDBaseConfigScaler):
    """
    Legacy linear scaler: every feature, including ``c_min``, divides by a shared ``10_000``.

    Uses :data:`ABCD_NMAX_BOUNDS`. ``c_min`` is not a ``c_min/c_max`` ratio here, so the base
    class's ``_scale_c_min``/``_unscale_c_min`` helpers are unused. The inverse is exact.
    """

    def __init__(self) -> None:
        """Initialise the scaler, building lo/hi tensors from :data:`ABCD_NMAX_BOUNDS`."""
        super().__init__()
        self._lo = torch.tensor(
            [ABCD_NMAX_BOUNDS[k][0] for k in ABCD_CONFIG_KEYS], dtype=torch.float32
        )
        self._hi = torch.tensor(
            [ABCD_NMAX_BOUNDS[k][1] for k in ABCD_CONFIG_KEYS], dtype=torch.float32
        )

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        return (x - self._lo) / (self._hi - self._lo)

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable (plain linear inverse map).

        :param x: Normalised float tensor of shape ``(..., 9)``.

        :returns: Tensor in the original feature scale, integer features left unrounded.
        """
        lo = self._lo.to(x.device)
        hi = self._hi.to(x.device)
        return x * (hi - lo) + lo


class ABCDLogConfigScaler(ABCDBaseConfigScaler):
    """
    Log-compresses size-like features against `n`.

    ``c_max``, ``d_min``, ``d_max``, ``nout`` as ``log1p(x) / log1p(n)``; `n` as
    ``log1p(n) / log1p(10_000)``. `t1`, `t2`, `xi`, `c_min` keep :class:`ABCDConfigScaler`'s
    treatment. The inverse is exact.
    """

    def __init__(self) -> None:
        """Initialise the scaler, caching feature indices and the linear bounds it still uses."""
        super().__init__()
        self._log_n_max = math.log1p(10_000.0)
        linear_keys = ("t1", "t2", "xi")
        self._linear_indices = [ABCD_CONFIG_KEYS.index(k) for k in linear_keys]
        self._linear_lo = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][0] for k in linear_keys], dtype=torch.float32
        )
        self._linear_hi = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][1] for k in linear_keys], dtype=torch.float32
        )
        self._n_idx = ABCD_CONFIG_KEYS.index("n")
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

        out[..., self._c_min_idx] = self._scale_c_min(x)
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable. Use :meth:`inverse_transform` for integer-rounded output.

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
        cols[self._c_min_idx] = self._unscale_c_min(x, cols[self._c_max_idx].clamp(min=1.0))
        return torch.stack(cols, dim=-1)


class ABCDRelativeConfigScaler(ABCDBaseConfigScaler):
    """
    Normalise ABCD config tensors to ``[0, 1]`` using hard-constraint anchors, not dataset bounds.

    - ``c_max/n``, ``d_max/n`` (``x <= n``).
    - ``d_min -> log(d_min)/log(d_max)`` (``d_min <= d_max``), i.e. the exponent ``s`` in
      ``d_min = d_max ** s``.
    - ``nout -> log1p(nout)/log1p(n)`` (``nout <= n``).
    - `n`, `t1`, `t2`, `xi`, `c_min` keep :class:`ABCDConfigScaler`'s treatment.

    The inverse is exact: `n`, then `d_max` before `d_min`, then `c_max` before `c_min`.
    """

    def __init__(self) -> None:
        """Initialise the scaler, caching feature indices and the linear bounds it still uses."""
        super().__init__()
        linear_keys = ("t1", "t2", "xi")
        self._linear_indices = [ABCD_CONFIG_KEYS.index(k) for k in linear_keys]
        self._linear_lo = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][0] for k in linear_keys], dtype=torch.float32
        )
        self._linear_hi = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][1] for k in linear_keys], dtype=torch.float32
        )
        self._n_idx = ABCD_CONFIG_KEYS.index("n")
        self._n_lo, self._n_hi = ABCD_PARAM_BOUNDS["n"]
        self._d_min_idx = ABCD_CONFIG_KEYS.index("d_min")
        self._d_max_idx = ABCD_CONFIG_KEYS.index("d_max")
        self._nout_idx = ABCD_CONFIG_KEYS.index("nout")

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        Scales size-like features against their hard-constraint anchors (`n`, `d_max`).

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

        out[..., self._c_min_idx] = self._scale_c_min(x)
        return out

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Recover the original scale from a normalised tensor, without integer rounding.

        Fully differentiable. Use :meth:`inverse_transform` for integer-rounded output.

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
        cols[self._c_min_idx] = self._unscale_c_min(x, c_max_raw.clamp(min=1.0))
        return torch.stack(cols, dim=-1)


class ABCDIdentityConfigScaler(ABCDBaseConfigScaler):
    """
    No-op scaler: every feature stays in raw scale.

    Gives the "no scaling" comparison arm a real scaler object (rather than ``None``) to attach
    to :class:`~dcba.training.loss.ABCDConstraintPenaltyLoss`, which always requires one.
    """

    def transform(self, x: Tensor) -> Tensor:
        """
        Return ``x`` unchanged.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: ``x``, unmodified.
        """
        return x

    def denormalise(self, x: Tensor) -> Tensor:
        """
        Return ``x`` unchanged.

        :param x: Float tensor of shape ``(..., 9)``.

        :returns: ``x``, unmodified.
        """
        return x


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
