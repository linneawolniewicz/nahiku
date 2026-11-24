"""Regression checks for diagnostic helpers and rejected public inputs."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from nahiku import Nahiku
from nahiku.balmung import Balmung


def test_balmung_negative_amplitude_has_equivalent_positive_representation():
    time = np.linspace(0, 10, 1001)
    flux = -0.5 * np.cos(2 * np.pi * 2 * time + 0.3)
    engine = Balmung(time, flux)
    theta = engine.fit([2.0, -0.5, 0.3])
    assert theta[1] > 0
    np.testing.assert_allclose(engine.model(time, *theta), flux, atol=1e-3)


def test_balmung_background_and_diagnostic_plots():
    time = np.linspace(0, 10, 1001)
    flux = 0.5 * np.cos(2 * np.pi * 2.3 * time + 0.4)
    engine = Balmung(time, flux)
    grid = np.geomspace(1, 10, 101)
    np.testing.assert_allclose(engine.estimate_background(grid, np.full(101, 3.0), 0.1), 3.0)
    engine.prewhiten(fmin=1, fmax=4, maxiter=1, minimum_snr=4, diagnose=True)
    assert len(engine.removed) == 1
    for method, expected in [
        (engine.plot_lc, engine.flux),
        (engine.plot_residual, engine.residual),
    ]:
        ax = method()
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), expected)
        plt.close(ax.figure)
        fig, ax = plt.subplots()
        assert method(ax=ax) is ax
        plt.close(fig)
    for supplied in [False, True]:
        fig, ax = plt.subplots() if supplied else (None, None)
        plotted = engine.plot(ax=ax)
        np.testing.assert_allclose(plotted.lines[1].get_xdata(), [engine.removed[0][0]])
        np.testing.assert_allclose(plotted.lines[1].get_ydata(), [engine.removed[0][1]])
        plt.close(plotted.figure)


@pytest.fixture(scope="module")
def fitted_small():
    return Nahiku(np.linspace(0, 4, 25), np.sin(np.linspace(0, 4, 25))).fit_gp(
        kernel="rbf", iterations=2
    )


@pytest.mark.parametrize(
    "options",
    [
        {"min_points": 0},
        {"step": 1.5},
        {"min_points": 5, "max_points": 3},
        {"max_points": 25},
        {"metric": "unknown"},
        {"gap_factor": 0},
        {"threshold": None, "top_k": 0},
        {"threshold": float("nan")},
        {"threshold": 1.1},
    ],
)
def test_exhaustive_invalid_settings_leave_existing_fit_intact(fitted_small, options):
    fit = fitted_small._fit
    with pytest.raises(ValueError):
        fitted_small.exhaustive_search(**options)
    assert fitted_small._fit is fit
    assert fitted_small.results == {}


@pytest.mark.parametrize(
    "time, errors",
    [([[0, 1]], None), ([0, np.nan], None), ([0, 1], [1]), ([0, 1], [0, 1]), ([0, 1], [1, np.inf])],
)
def test_prediction_rejects_invalid_grid_and_errors(fitted_small, time, errors):
    with pytest.raises(ValueError, match="Prediction"):
        fitted_small.predict(time, flux_err=errors)


@pytest.mark.parametrize(
    "options",
    [
        {"samples_per_peak": True},
        {"samples_per_peak": 2.5},
        {"method": "weighted", "samples_per_peak": 1},
        {"method": "weighted", "fit_mask": [True, True]},
        {"method": "weighted", "fit_mask": np.zeros(25, bool)},
        {"fmin": 5, "fmax": 1},
        {"minimum_snr": 0},
        {"max_components": 0},
    ],
)
def test_prewhitening_rejects_invalid_settings_without_mutation(options):
    lc = Nahiku(np.linspace(0, 10, 25), np.sin(np.linspace(0, 10, 25)))
    flux = lc.flux.copy()
    with pytest.raises(ValueError):
        lc.prewhiten(**({"fmin": 1, "fmax": 5} | options))
    np.testing.assert_array_equal(lc.flux, flux)
    assert lc.history == [] and lc.prewhitening is None


@pytest.mark.parametrize("width", [0, -1, np.inf, np.nan])
def test_invalid_bin_width_preserves_parent(width):
    lc = Nahiku(np.arange(6.0), np.arange(6.0))
    with pytest.raises(ValueError, match="Bin width"):
        lc.bin(width)
    assert lc.history == []


def test_constant_flux_standardization_is_rejected_without_mutation():
    lc = Nahiku(np.arange(6.0), np.ones(6))
    with pytest.raises(ValueError, match="constant flux"):
        lc.standardize()
    np.testing.assert_array_equal(lc.flux, lc.original_flux)
    assert lc.history == []
