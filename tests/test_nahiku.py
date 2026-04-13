import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from nahiku import Nahiku


def test_nahiku_init():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    nahiku = Nahiku(x, y)
    assert len(nahiku.time) == 100
    assert len(nahiku.flux) == 100
    assert "true" in nahiku.anomalies


def test_nahiku_synthetic_noise():
    nahiku = Nahiku.from_synthetic_parameterized_noise(num_days=10, num_steps=100)
    assert len(nahiku.time) == 100
    assert nahiku.dominant_period > 0


def test_nahiku_synthetic_gp():
    # num_steps should be num_days / cadence or similar, let's just use defaults
    nahiku = Nahiku.from_synthetic_parameterized_gp(num_days=5, num_steps=50)
    assert len(nahiku.time) == 50
    assert nahiku.dominant_period > 0


def test_nahiku_synthetic_gp_residuals():
    # Test high residuals addition
    nahiku = Nahiku.from_synthetic_parameterized_gp(
        num_days=5, num_steps=50, add_high_residuals=True, num_high_residuals=5
    )
    assert len(nahiku.time) == 50


def test_nahiku_prewhiten():
    x = np.linspace(0, 10, 100)
    y = 10 * np.sin(x * 2 * np.pi / 2.0) + np.random.normal(0, 0.01, 100)
    nahiku = Nahiku(x, y)
    initial_flux = np.copy(nahiku.flux)
    nahiku.prewhiten(plot=False, maxiter=2, minimum_snr=0)
    assert not np.allclose(nahiku.flux, initial_flux)


def test_nahiku_inject_anomaly():
    x = np.linspace(0, 10, 100)
    y = np.random.normal(0, 0.01, 100)
    nahiku = Nahiku(x, y)
    # Test multiple shapes
    nahiku.inject_anomaly(
        1, absolute_width=0.5, absolute_depth=5, shapes=["exocomet"], idxs=[30]
    )
    nahiku.inject_anomaly(
        1, absolute_width=0.5, absolute_depth=5, shapes=["gaussian"], idxs=[60]
    )
    nahiku.inject_anomaly(
        1, absolute_width=0.5, absolute_depth=5, shapes=["tophat"], idxs=[80]
    )
    assert 30 in nahiku.anomalies["injected"]
    assert 60 in nahiku.anomalies["injected"]
    assert 80 in nahiku.anomalies["injected"]


def test_nahiku_greedy_search():
    x = np.linspace(0, 5, 50)
    y = np.random.normal(0, 0.1, 50)
    y[20:25] -= 10
    nahiku = Nahiku(x, y)
    res = nahiku.greedy_search(plot=False, training_iterations=5)
    assert res is not None


def test_nahiku_exhaustive_search():
    x = np.linspace(0, 2, 20)
    y = np.random.normal(0, 0.001, 20)
    y[10:12] -= 10.0
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, max_anomaly_len=3, training_iterations=5, plot=False
    )
    assert res is not None


def test_nahiku_standardize():
    x = np.arange(10)
    y = np.arange(10) * 10.0 + 5.0
    nahiku = Nahiku(x, y)
    # The constructor calls get_dominant_period which calls standardize if std != 1
    # So it should already be roughly standardized
    assert np.isclose(np.std(nahiku.flux), 1.0, atol=0.1)
    assert np.isclose(np.mean(nahiku.flux), 0.0, atol=0.1)


@patch("lightkurve.search_lightcurve")
def test_from_lightkurve(mock_search):
    # Mock the lightkurve download chain
    mock_lc = MagicMock()
    # Create time and flux that will NOT result in empty slope < 0
    # Actually, a simple way is to ensure freqs and power have some length
    mock_lc.time.value = np.linspace(0, 10, 100)
    # Give it some signal so periodogram isn't flat
    mock_lc.flux.value = np.sin(mock_lc.time.value) + 1.0

    mock_search.return_value.download_all.return_value.stitch.return_value.remove_nans.return_value = (
        mock_lc
    )

    nahiku = Nahiku.from_lightkurve(target="test", mission="TESS")
    assert len(nahiku.time) == 100


def test_check_identified_anomalies():
    x = np.linspace(0, 10, 100)
    # Give it some signal
    y = np.sin(x)
    nahiku = Nahiku(x, y)
    nahiku.anomalies["injected"] = [50]
    nahiku.anomalies["identified"] = [51]
    # This just prints things, but let's make sure it doesn't crash
    nahiku.check_identified_anomalies(buffer=5)
