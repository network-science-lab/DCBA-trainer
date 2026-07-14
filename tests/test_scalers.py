"""Unit tests for ABCD config scalers."""

import math

import torch

from dcba.dataset.transforms import (
    ABCD_CONFIG_KEYS,
    ABCDConfigScaler,
    ABCDLogConfigScaler,
    ABCDRelativeConfigScaler,
)

_N_MAX = 10_000


def _raw_batch() -> torch.Tensor:
    """A small, constraint-respecting batch of raw ABCD configs, varied in scale.

    Columns follow :data:`ABCD_CONFIG_KEYS`: ``[n, t1, t2, xi, c_min, c_max, d_min, d_max, nout]``.
    Each row keeps ``c_min <= c_max <= n``, ``d_min <= d_max <= n`` and ``nout <= n``.
    """
    return torch.tensor(
        [
            [5000.0, 2.5, 3.1, 0.3, 20.0, 300.0, 2.0, 50.0, 15.0],
            [10000.0, 1.5, 1.8, 0.1, 5.0, 4000.0, 1.0, 30.0, 200.0],
            [50.0, 3.0, 2.0, 0.5, 2.0, 10.0, 1.0, 5.0, 1.0],
        ]
    )


class TestABCDLogConfigScalerRoundTrip:
    """The scaler's inverse must exactly recover the original raw values."""

    def test_round_trip_recovers_raw_values(self) -> None:
        """Transform then inverse_transform returns the original batch (up to int rounding)."""
        raw = _raw_batch()
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        recovered = scaler.inverse_transform(scaler.transform(raw))
        assert torch.allclose(recovered, raw, atol=1e-3)

    def test_round_trip_single_row(self) -> None:
        """Round-trip also holds for a single (non-batched) config row."""
        raw = _raw_batch()[0]
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        recovered = scaler.inverse_transform(scaler.transform(raw))
        assert torch.allclose(recovered, raw, atol=1e-3)


class TestABCDLogConfigScalerRange:
    """Scaled output must stay within the model's [0, 1] input/output domain."""

    def test_scaled_values_in_unit_range(self) -> None:
        """Every scaled feature lies in [0, 1] for constraint-respecting raw inputs."""
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        scaled = scaler.transform(_raw_batch())
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0 + 1e-6

    def test_n_matches_closed_form(self) -> None:
        """`n` is scaled as log1p(n) / log1p(n_max), independent of the other features."""
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        n_idx = ABCD_CONFIG_KEYS.index("n")
        expected = torch.log1p(raw[:, n_idx]) / math.log1p(_N_MAX)
        assert torch.allclose(scaled[:, n_idx], expected, atol=1e-6)

    def test_c_min_still_scaled_as_ratio_to_c_max(self) -> None:
        """c_min keeps ABCDConfigScaler's `c_min / c_max` treatment, unaffected by the log change."""
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        c_max_idx = ABCD_CONFIG_KEYS.index("c_max")
        expected = raw[:, c_min_idx] / raw[:, c_max_idx]
        assert torch.allclose(scaled[:, c_min_idx], expected, atol=1e-6)


class TestABCDLogConfigScalerRelativeToOwnN:
    """The log-relative features must depend on each graph's own n, not a fixed bound."""

    def test_same_raw_value_scales_differently_under_different_n(self) -> None:
        """Equal raw d_max values scale to different numbers when their graphs' n differ."""
        scaler = ABCDLogConfigScaler(n_max=_N_MAX)
        small_n = torch.tensor([[200.0, 2.0, 2.0, 0.2, 2.0, 20.0, 1.0, 20.0, 1.0]])
        large_n = torch.tensor([[9000.0, 2.0, 2.0, 0.2, 2.0, 20.0, 1.0, 20.0, 1.0]])
        d_max_idx = ABCD_CONFIG_KEYS.index("d_max")
        scaled_small = scaler.transform(small_n)[0, d_max_idx]
        scaled_large = scaler.transform(large_n)[0, d_max_idx]
        assert scaled_small > scaled_large


class TestABCDLogConfigScalerRangeUtilisation:
    """The whole point of the log-relative scaler: better use of [0, 1] for typical values."""

    def test_better_spread_than_fixed_bound_scaler_for_typical_values(self) -> None:
        """d_max values that are a small fraction of n_max spread out more under the log scaler.

        `ABCDConfigScaler` maps `d_max` linearly against a fixed bound observed on one dataset;
        typical graphs (`d_max` a small fraction of that bound) collapse near 0 (see
        `scripts/check_config_scaling.py`). `ABCDLogConfigScaler` maps `d_max` relative to each
        graph's own n in log space, which should spread the same values out much more.
        """
        raw = torch.tensor(
            [
                [5000.0, 2.0, 2.0, 0.2, 2.0, 20.0, 1.0, 20.0, 5.0],
                [5000.0, 2.0, 2.0, 0.2, 2.0, 20.0, 1.0, 30.0, 5.0],
                [5000.0, 2.0, 2.0, 0.2, 2.0, 20.0, 1.0, 45.0, 5.0],
            ]
        )
        d_max_idx = ABCD_CONFIG_KEYS.index("d_max")

        old_scaled = ABCDConfigScaler(n_max=_N_MAX).transform(raw)[:, d_max_idx]
        new_scaled = ABCDLogConfigScaler(n_max=_N_MAX).transform(raw)[:, d_max_idx]

        assert new_scaled.std() > old_scaled.std()
        assert old_scaled.max() < 0.01  # the starved-range problem this scaler fixes


class TestABCDRelativeConfigScalerRoundTrip:
    """The relative scaler's inverse must exactly recover the original raw values."""

    def test_round_trip_recovers_raw_values(self) -> None:
        """Transform then inverse_transform returns the original batch (up to int rounding)."""
        raw = _raw_batch()
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        recovered = scaler.inverse_transform(scaler.transform(raw))
        assert torch.allclose(recovered, raw, atol=1e-3)

    def test_round_trip_single_row(self) -> None:
        """Round-trip also holds for a single (non-batched) config row."""
        raw = _raw_batch()[0]
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        recovered = scaler.inverse_transform(scaler.transform(raw))
        assert torch.allclose(recovered, raw, atol=1e-3)

    def test_round_trip_nout_zero(self) -> None:
        """Nout = 0 (a legal value -- the parameter is optional) survives the round trip."""
        raw = _raw_batch()[0]
        raw[ABCD_CONFIG_KEYS.index("nout")] = 0.0
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        recovered = scaler.inverse_transform(scaler.transform(raw))
        assert torch.allclose(recovered, raw, atol=1e-3)


class TestABCDRelativeConfigScalerSemantics:
    """Each feature must follow its documented hard-constraint-anchored closed form."""

    def test_scaled_values_in_unit_range(self) -> None:
        """Every scaled feature lies in [0, 1] for constraint-respecting raw inputs."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        scaled = scaler.transform(_raw_batch())
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0 + 1e-6

    def test_c_max_and_d_max_are_fractions_of_n(self) -> None:
        """c_max and d_max are scaled as plain ratios to the same row's n."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        n = raw[:, ABCD_CONFIG_KEYS.index("n")]
        for key in ("c_max", "d_max"):
            i = ABCD_CONFIG_KEYS.index(key)
            assert torch.allclose(scaled[:, i], raw[:, i] / n, atol=1e-6)

    def test_d_min_is_log_exponent_of_d_max(self) -> None:
        """d_min is scaled as the exponent s with d_min = d_max ** s."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        d_min = raw[:, ABCD_CONFIG_KEYS.index("d_min")]
        d_max = raw[:, ABCD_CONFIG_KEYS.index("d_max")]
        expected = torch.log(d_min) / torch.log(d_max)
        assert torch.allclose(scaled[:, ABCD_CONFIG_KEYS.index("d_min")], expected, atol=1e-6)

    def test_nout_matches_log_closed_form(self) -> None:
        """Nout is scaled as log1p(nout) / log1p(n)."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        n = raw[:, ABCD_CONFIG_KEYS.index("n")]
        nout = raw[:, ABCD_CONFIG_KEYS.index("nout")]
        expected = torch.log1p(nout) / torch.log1p(n)
        assert torch.allclose(scaled[:, ABCD_CONFIG_KEYS.index("nout")], expected, atol=1e-6)

    def test_c_min_still_scaled_as_ratio_to_c_max(self) -> None:
        """c_min keeps the `c_min / c_max` treatment the constraint loss relies on."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        raw = _raw_batch()
        scaled = scaler.transform(raw)
        c_min_idx = ABCD_CONFIG_KEYS.index("c_min")
        c_max_idx = ABCD_CONFIG_KEYS.index("c_max")
        expected = raw[:, c_min_idx] / raw[:, c_max_idx]
        assert torch.allclose(scaled[:, c_min_idx], expected, atol=1e-6)


class TestABCDRelativeConfigScalerNInvariance:
    """The point of relative scaling: equal fractions of n scale identically across graph sizes."""

    def test_equal_fractions_scale_identically(self) -> None:
        """Two graphs with c_max/n and d_max/n equal get identical scaled values despite n 10x."""
        scaler = ABCDRelativeConfigScaler(n_max=_N_MAX)
        small = torch.tensor([[500.0, 2.0, 2.0, 0.2, 10.0, 100.0, 2.0, 50.0, 5.0]])
        large = torch.tensor([[5000.0, 2.0, 2.0, 0.2, 100.0, 1000.0, 2.0, 500.0, 50.0]])
        s_small = scaler.transform(small)[0]
        s_large = scaler.transform(large)[0]
        for key in ("c_max", "d_max"):
            i = ABCD_CONFIG_KEYS.index(key)
            assert torch.isclose(s_small[i], s_large[i], atol=1e-6)
