"""Shoulder fallback checks using real timestamp-aware periodograms."""

import numpy as np
import pytest
from nahiku import Nahiku


def test_shoulder_is_explicit_and_preserves_rejected_peak_diagnostics():
    rng = np.random.default_rng(12)
    time = np.sort(rng.uniform(0, 10, 100))
    lc = Nahiku(time, rng.normal(size=len(time)))
    settings = dict(minimum_period=1, maximum_period=5, maximum_fap=1e-200)
    assert lc.estimate_period(**settings) is None
    rejected = lc.period_result.copy()
    period = lc.estimate_period(**settings, fallback="shoulder")
    result = lc.period_result
    assert 1 <= period <= 5
    assert result["fallback_used"] and not result["peak_accepted"]
    assert result["reason"] == "shoulder_fallback"
    assert result["false_alarm_probability"] == rejected["false_alarm_probability"]
    assert result["candidate_period"] == rejected["candidate_period"]
    np.testing.assert_array_equal(result["power"], rejected["power"])
    assert result["spectral_slope"][result["shoulder_index"]] < 0
    np.testing.assert_array_equal(lc.flux, lc.original_flux)


def test_shoulder_cap_uses_duration_not_absolute_timestamp():
    rng = np.random.default_rng(12)
    time = np.sort(rng.uniform(0, 10, 100))
    settings = dict(minimum_period=5, maximum_period=100, maximum_fap=1e-200,
                    fallback="shoulder")
    relative = Nahiku(time, time).estimate_period(**settings)
    lc = Nahiku(time + 2457000, time)
    period = lc.estimate_period(**settings)
    assert lc.period_result["shoulder_period_uncapped"] > np.ptp(time)
    assert period == pytest.approx(np.ptp(time))
    assert period == pytest.approx(relative)


def test_rising_spectrum_has_no_shoulder():
    time = np.linspace(0, 10, 101)
    lc = Nahiku(time, np.sin(2 * np.pi * time / 2))
    assert lc.estimate_period(minimum_period=2.1, maximum_period=2.5,
                              fallback="shoulder") is None
    assert lc.period_result["reason"] == "no_shoulder"
    assert not lc.period_result["fallback_used"]


def test_capped_shoulder_must_respect_requested_lower_bound():
    time = np.linspace(0, 10, 101)
    lc = Nahiku(time, time)
    assert lc.estimate_period(minimum_period=20, maximum_period=100,
                              maximum_fap=1e-200, fallback="shoulder") is None
    assert lc.period_result["reason"] == "shoulder_outside_bounds"


def test_accepted_peak_takes_precedence_over_fallback():
    rng = np.random.default_rng(8)
    time = np.sort(rng.uniform(0, 30, 500))
    lc = Nahiku(time, np.sin(2 * np.pi * time / 2.35) + rng.normal(0, .03, len(time)))
    period = lc.estimate_period(minimum_period=1, maximum_period=5, fallback="shoulder")
    assert period == pytest.approx(2.35, rel=.01)
    assert lc.period_result["peak_accepted"]
    assert not lc.period_result["fallback_used"]
    assert lc.period_result["reason"] == "accepted"


def test_constant_flux_and_invalid_fallback():
    lc = Nahiku(np.arange(10.), np.ones(10))
    assert lc.estimate_period(fallback="shoulder") is None
    assert lc.period_result["reason"] == "constant_flux"
    assert not lc.period_result["fallback_used"]
    with pytest.raises(ValueError, match="fallback"):
        lc.estimate_period(fallback="anything")
