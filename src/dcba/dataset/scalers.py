"""Scalers and theta-vector transforms for ABCD config records."""

import math

import torch
from dcba_data_set.graph_io.data_models import DCBAInstanceConfig
from pydantic import BaseModel
from torch import Tensor
from torch_geometric.transforms import BaseTransform

#: Ordered list of numerical ABCD config keys used as model input features.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]
ABCD_CONFIG_IDX: dict[str, int] = {name: i for i, name in enumerate(ABCD_CONFIG_KEYS)}

# Indices of features scaled differently
ABCD_INT_FEATURE_INDICES: list[int] = [0, 4, 5, 6, 7, 8]
ABCD_CONFIG_LINEAR_IDX: list[int] = [ABCD_CONFIG_IDX[k] for k in ("t1", "t2", "xi")]
ABCD_CONFIG_LOG_IDX: list[int] = [ABCD_CONFIG_IDX[k] for k in ("c_max", "d_max", "nout")]


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
#: bound -- are chosen ceilings, several observed on ``abcd-big``. ``c_min`` and ``d_min`` are
#: bounded ``[0, 1]`` since :class:`ABCDEmpiricalConfigScaler` scales them against ``c_max`` and
#: ``d_max`` rather than a fixed ceiling.
ABCD_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "n": (1.0, 10_000.0),  # @assert n > 0; upper bound is a chosen ceiling
    "t1": (1.0, 5.0),  # @assert alpha >= 1; upper bound is a chosen ceiling
    "t2": (1.0, 5.0),  # @assert alpha >= 1; upper bound is a chosen ceiling
    "xi": (0.0, 1.0),  # 0 <= xi <= 1 (hard constraint)
    "c_min": (0.0, 1.0),  # fraction of c_max -- see ABCDEmpiricalConfigScaler
    "c_max": (1.0, 6_000.0),  # observed max ~4938 on abcd-big
    "d_min": (0.0, 1.0),  # fraction of d_max -- see ABCDEmpiricalConfigScaler
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


def _scale_ratio(x: Tensor, key: str, anchor_key: str) -> Tensor:
    """
    Compute ``key`` as a fraction of its raw anchor (clamped >= 1 to avoid a div by zero).

    :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.
    :param key: Feature to scale, e.g. ``"c_min"``.
    :param anchor_key: Feature to scale it against, e.g. ``"c_max"``.

    :returns: Normalised ``key`` column, shape ``(...,)``.
    """
    anchor_raw = x[..., ABCD_CONFIG_IDX[anchor_key]].clamp(min=1.0)
    return x[..., ABCD_CONFIG_IDX[key]] / anchor_raw


def _unscale_ratio(x: Tensor, key: str, anchor_raw: Tensor) -> Tensor:
    """
    Recover raw ``key`` from its ``key / anchor`` ratio.

    :param x: Normalised float tensor of shape ``(..., 9)`` (only its ``key`` column is read).
    :param key: Feature to recover, e.g. ``"c_min"``.
    :param anchor_raw: Already-recovered raw anchor. Clamp before calling if needed; not applied
        here.

    :returns: Raw ``key`` column, shape ``(...,)``.
    """
    return x[..., ABCD_CONFIG_IDX[key]] * anchor_raw


def _scale_log_ratio(x: Tensor, key: str, anchor_key: str) -> Tensor:
    """
    Compute ``key`` as the exponent ``s`` in ``key = anchor ** s``.

    Anchor clamped >= 2 so its log is positive; only reachable on degenerate raw configs.

    :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.
    :param key: Feature to scale, e.g. ``"d_min"``.
    :param anchor_key: Feature to scale it against, e.g. ``"d_max"``.

    :returns: Normalised ``key`` column, shape ``(...,)``.
    """
    return torch.log(x[..., ABCD_CONFIG_IDX[key]].clamp(min=1.0)) / torch.log(
        x[..., ABCD_CONFIG_IDX[anchor_key]].clamp(min=2.0)
    )


def _unscale_log_ratio(x: Tensor, key: str, anchor_raw: Tensor) -> Tensor:
    """
    Recover raw ``key`` from its ``log(key) / log(anchor)`` exponent.

    :param x: Normalised float tensor of shape ``(..., 9)`` (only its ``key`` column is read).
    :param key: Feature to recover, e.g. ``"d_min"``.
    :param anchor_raw: Already-recovered raw anchor, clamped >= 2 here.

    :returns: Raw ``key`` column, shape ``(...,)``.
    """
    return anchor_raw.clamp(min=2.0) ** x[..., ABCD_CONFIG_IDX[key]]


class ABCDBaseConfigScaler:
    """
    Shared scaffolding for the ``[0, 1]`` ABCD config scalers below.

    Provides the integer-rounding :meth:`inverse_transform` and :meth:`__call__`, used by every
    scaler. Subclasses implement :meth:`transform`/:meth:`denormalise`; the ``c_min``/``d_min``
    anchoring helpers used by some of them are module-level functions (:func:`_scale_ratio` etc.),
    not part of this base class, since not every scaler needs them.
    """

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


class ABCDEmpiricalConfigScaler(ABCDBaseConfigScaler):
    """
    Normalise and denormalise ABCD config tensors feature-wise to ``[0, 1]``.

    Linear (min-max) map per feature from :data:`ABCD_PARAM_BOUNDS`, except ``c_min`` and
    ``d_min``, scaled as fractions of their true constraint partners: ``c_min / c_max`` and
    ``d_min / d_max``. The inverse is exact.
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
        out[..., ABCD_CONFIG_IDX["d_min"]] = _scale_ratio(x, "d_min", "d_max")
        out[..., ABCD_CONFIG_IDX["c_min"]] = _scale_ratio(x, "c_min", "c_max")
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
        # d_max/c_max must already be in their final (raw) scale before d_min/c_min can be
        # recovered from them.
        cols[ABCD_CONFIG_IDX["d_min"]] = _unscale_ratio(x, "d_min", cols[ABCD_CONFIG_IDX["d_max"]])
        cols[ABCD_CONFIG_IDX["c_min"]] = _unscale_ratio(x, "c_min", cols[ABCD_CONFIG_IDX["c_max"]])
        return torch.stack(cols, dim=-1)


class ABCDNMaxConfigScaler(ABCDBaseConfigScaler):
    """
    Legacy linear scaler: every feature divides by a shared ``10_000``.

    Uses :data:`ABCD_NMAX_BOUNDS`. ``c_min`` and ``d_min`` are not anchored to their constraint
    partners here, so the :func:`_scale_ratio` helpers are unused. The inverse is exact.
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

    ``c_max``, ``d_max``, ``nout`` as ``log1p(x) / log1p(n)``; `n` as
    ``log1p(n) / log1p(10_000)``; `d_min` as ``log(d_min) / log(d_max)`` (its true constraint
    partner). `t1`, `t2`, `xi`, `c_min` keep :class:`ABCDEmpiricalConfigScaler`'s treatment.
    The inverse is exact.
    """

    def __init__(self) -> None:
        """Initialise the scaler, caching the linear bounds it still uses."""
        super().__init__()
        self._log_n_max = math.log1p(10_000.0)
        linear_keys = ("t1", "t2", "xi")
        self._linear_lo = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][0] for k in linear_keys], dtype=torch.float32
        )
        self._linear_hi = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][1] for k in linear_keys], dtype=torch.float32
        )

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature, log-compressing size-like features against `n`.

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = torch.zeros_like(x)
        n_raw = x[..., ABCD_CONFIG_IDX["n"]].clamp(min=1.0)
        log_n = torch.log1p(n_raw)

        out[..., ABCD_CONFIG_IDX["n"]] = log_n / self._log_n_max
        out[..., ABCD_CONFIG_LINEAR_IDX] = (x[..., ABCD_CONFIG_LINEAR_IDX] - self._linear_lo) / (
            self._linear_hi - self._linear_lo
        )
        out[..., ABCD_CONFIG_LOG_IDX] = torch.log1p(
            x[..., ABCD_CONFIG_LOG_IDX].clamp(min=0.0)
        ) / log_n.unsqueeze(-1)

        out[..., ABCD_CONFIG_IDX["d_min"]] = _scale_log_ratio(x, "d_min", "d_max")
        out[..., ABCD_CONFIG_IDX["c_min"]] = _scale_ratio(x, "c_min", "c_max")
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
        n_raw = torch.expm1(x[..., ABCD_CONFIG_IDX["n"]] * self._log_n_max).clamp(min=1.0)
        log_n = torch.log1p(n_raw)
        lin = x[..., ABCD_CONFIG_LINEAR_IDX] * (hi - lo) + lo
        logs = torch.expm1(x[..., ABCD_CONFIG_LOG_IDX] * log_n.unsqueeze(-1))

        cols: list[Tensor] = [torch.empty(0)] * len(ABCD_CONFIG_KEYS)
        cols[ABCD_CONFIG_IDX["n"]] = n_raw
        for j, i in enumerate(ABCD_CONFIG_LINEAR_IDX):
            cols[i] = lin[..., j]
        for j, i in enumerate(ABCD_CONFIG_LOG_IDX):
            cols[i] = logs[..., j]
        cols[ABCD_CONFIG_IDX["d_min"]] = _unscale_log_ratio(
            x, "d_min", cols[ABCD_CONFIG_IDX["d_max"]]
        )
        cols[ABCD_CONFIG_IDX["c_min"]] = _unscale_ratio(
            x, "c_min", cols[ABCD_CONFIG_IDX["c_max"]].clamp(min=1.0)
        )
        return torch.stack(cols, dim=-1)


class ABCDRelativeConfigScaler(ABCDBaseConfigScaler):
    """
    Normalise ABCD config tensors to ``[0, 1]`` using hard-constraint anchors, not dataset bounds.

    - ``c_max/n``, ``d_max/n`` (``x <= n``).
    - ``d_min -> log(d_min)/log(d_max)`` (``d_min <= d_max``), i.e. the exponent ``s`` in
      ``d_min = d_max ** s``.
    - ``nout -> log1p(nout)/log1p(n)`` (``nout <= n``).
    - `n`, `t1`, `t2`, `xi`, `c_min` keep :class:`ABCDEmpiricalConfigScaler`'s treatment.

    The inverse is exact: `n`, then `d_max` before `d_min`, then `c_max` before `c_min`.
    """

    def __init__(self) -> None:
        """Initialise the scaler, caching the linear bounds it still uses."""
        super().__init__()
        linear_keys = ("t1", "t2", "xi")
        self._linear_lo = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][0] for k in linear_keys], dtype=torch.float32
        )
        self._linear_hi = torch.tensor(
            [ABCD_PARAM_BOUNDS[k][1] for k in linear_keys], dtype=torch.float32
        )
        self._n_lo, self._n_hi = ABCD_PARAM_BOUNDS["n"]

    def transform(self, x: Tensor) -> Tensor:
        """
        Normalise ``x`` to ``[0, 1]`` per feature.

        Scales size-like features against their hard-constraint anchors (`n`, `d_max`).

        :param x: Float tensor of shape ``(..., 9)``, raw (unscaled) feature values.

        :returns: Normalised tensor of the same shape.
        """
        out = torch.zeros_like(x)
        n_raw = x[..., ABCD_CONFIG_IDX["n"]].clamp(min=1.0)

        out[..., ABCD_CONFIG_IDX["n"]] = (n_raw - self._n_lo) / (self._n_hi - self._n_lo)
        out[..., ABCD_CONFIG_LINEAR_IDX] = (x[..., ABCD_CONFIG_LINEAR_IDX] - self._linear_lo) / (
            self._linear_hi - self._linear_lo
        )

        out[..., ABCD_CONFIG_IDX["c_max"]] = x[..., ABCD_CONFIG_IDX["c_max"]] / n_raw
        out[..., ABCD_CONFIG_IDX["d_max"]] = x[..., ABCD_CONFIG_IDX["d_max"]] / n_raw
        out[..., ABCD_CONFIG_IDX["nout"]] = torch.log1p(
            x[..., ABCD_CONFIG_IDX["nout"]].clamp(min=0.0)
        ) / torch.log1p(n_raw)

        out[..., ABCD_CONFIG_IDX["d_min"]] = _scale_log_ratio(x, "d_min", "d_max")
        out[..., ABCD_CONFIG_IDX["c_min"]] = _scale_ratio(x, "c_min", "c_max")
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
        n_raw = (x[..., ABCD_CONFIG_IDX["n"]] * (self._n_hi - self._n_lo) + self._n_lo).clamp(
            min=1.0
        )
        lin = x[..., ABCD_CONFIG_LINEAR_IDX] * (hi - lo) + lo

        cols: list[Tensor] = [torch.empty(0)] * len(ABCD_CONFIG_KEYS)
        cols[ABCD_CONFIG_IDX["n"]] = n_raw
        for j, i in enumerate(ABCD_CONFIG_LINEAR_IDX):
            cols[i] = lin[..., j]

        c_max_raw = x[..., ABCD_CONFIG_IDX["c_max"]] * n_raw
        d_max_raw = x[..., ABCD_CONFIG_IDX["d_max"]] * n_raw
        cols[ABCD_CONFIG_IDX["c_max"]] = c_max_raw
        cols[ABCD_CONFIG_IDX["d_max"]] = d_max_raw
        cols[ABCD_CONFIG_IDX["nout"]] = torch.expm1(
            x[..., ABCD_CONFIG_IDX["nout"]] * torch.log1p(n_raw)
        )
        cols[ABCD_CONFIG_IDX["d_min"]] = _unscale_log_ratio(x, "d_min", d_max_raw)
        cols[ABCD_CONFIG_IDX["c_min"]] = _unscale_ratio(x, "c_min", c_max_raw.clamp(min=1.0))
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
