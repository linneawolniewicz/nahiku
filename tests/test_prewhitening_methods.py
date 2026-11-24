"""Selectable prewhitening engines retain numerical and state guarantees."""

import numpy as np
import pytest
from nahiku import Nahiku
from nahiku.balmung import Balmung


@pytest.fixture
def pulsation():
    time = np.linspace(0, 10, 1001)
    flux = 10 + .5 * np.cos(2 * np.pi * 7.3 * time + .4)
    return Nahiku(time + 2457000, flux, np.full(len(time), .01))


@pytest.mark.parametrize("method", ["balmung", "weighted"])
def test_methods_recover_pulsation_and_preserve_continuum(pulsation, method):
    original = pulsation.flux.copy()
    errors = pulsation.flux_err.copy()
    result = pulsation.prewhiten(fmin=5, fmax=10, max_components=1, method=method)
    assert result.method == method
    assert result.frequencies[0] == pytest.approx(7.3, abs=.001)
    assert result.amplitudes[0] == pytest.approx(.5, abs=.001)
    assert np.std(result.residuals) < .001
    assert np.mean(result.residuals) == pytest.approx(10, abs=.001)
    np.testing.assert_allclose(result.residuals + result.model, original)
    np.testing.assert_allclose(result.model, result.amplitudes[0] * np.cos(
        2*np.pi*result.frequencies[0]*(pulsation.time-result.time_origin) + result.phases[0]))
    np.testing.assert_array_equal(pulsation.original_flux, original)
    np.testing.assert_array_equal(pulsation.flux_err, errors)
    assert pulsation.history[-1]["method"] == method


def test_default_delegates_to_balmung(pulsation):
    original = pulsation.flux.copy()
    engine = Balmung(pulsation.time - pulsation.time[0], original)
    engine.prewhiten(fmin=5, fmax=10, maxiter=2, minimum_snr=4,
                    diagnose=False)
    result = pulsation.prewhiten(fmin=5, fmax=10, max_components=2)
    assert result.method == "balmung"
    np.testing.assert_allclose(np.c_[result.frequencies, result.amplitudes, result.phases],
                               np.asarray(engine.removed))
    np.testing.assert_allclose(result.residuals, engine.residual + np.median(original))
    assert result.stopping_reason == ("component_limit" if len(engine.removed) >= 2 else "minimum_snr")


def test_invalid_options_leave_state_unchanged(pulsation):
    original = pulsation.flux.copy()
    for options in [dict(method="unknown"), dict(fit_mask=np.ones(len(original), bool))]:
        with pytest.raises(ValueError):
            pulsation.prewhiten(fmin=5, fmax=10, **options)
        np.testing.assert_array_equal(pulsation.flux, original)
        assert pulsation.history == []


def test_balmung_boundary_peak_is_a_frequency():
    engine = Balmung(np.arange(5.), np.arange(5.))
    assert engine.find_highest_peak(np.array([5., 6., 7.]), np.array([100., 2., 1.])) == 5.
    assert engine.find_highest_peak(np.array([5., 6., 7.]), np.array([1., 2., 100.])) == 7.


def test_balmung_no_components_and_invalidation(pulsation):
    pulsation.fit_gp(kernel="rbf", iterations=2)
    pulsation.greedy_search()
    result = pulsation.prewhiten(fmin=5, fmax=10, minimum_snr=1e12)
    assert len(result.frequencies) == 0
    assert result.stopping_reason == "minimum_snr"
    np.testing.assert_array_equal(result.model, np.zeros(len(pulsation.time)))
    assert pulsation.results == {}
    with pytest.raises(RuntimeError, match="fit_gp"):
        pulsation.predict()


def test_balmung_constant_flux():
    lc = Nahiku(np.linspace(0, 10, 100), np.ones(100)*12)
    result = lc.prewhiten(fmin=1, fmax=5)
    assert len(result.frequencies) == 0
    np.testing.assert_array_equal(result.residuals, lc.original_flux)
