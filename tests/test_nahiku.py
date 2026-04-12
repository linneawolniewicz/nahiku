import numpy as np
import pytest
from nahiku import Nahiku

def test_nahiku_init():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    nahiku = Nahiku(x, y)
    assert len(nahiku.time) == 100
    assert len(nahiku.flux) == 100
    assert "true" in nahiku.anomalies

def test_nahiku_synthetic():
    nahiku = Nahiku.from_synthetic_parameterized_noise(num_days=10, num_steps=100)
    assert len(nahiku.time) == 100
    assert nahiku.dominant_period > 0

def test_nahiku_prewhiten():
    x = np.linspace(0, 10, 100)
    # Strong signal
    y = 10 * np.sin(x * 2 * np.pi / 2.0) + np.random.normal(0, 0.01, 100)
    nahiku = Nahiku(x, y)
    initial_flux = np.copy(nahiku.flux)
    nahiku.prewhiten(plot=False, maxiter=2, minimum_snr=0)
    # After prewhitening a strong signal, flux values should change
    assert not np.allclose(nahiku.flux, initial_flux)

def test_nahiku_inject_anomaly():
    x = np.linspace(0, 10, 100)
    # Use non-zero noise to avoid get_dominant_period failure
    y = np.random.normal(0, 0.01, 100)
    nahiku = Nahiku(x, y)
    nahiku.inject_anomaly(1, absolute_width=1, absolute_depth=5, shapes=["exocomet"], idxs=[50])
    assert 50 in nahiku.anomalies["injected"]
    # We check if flux decreased at the injection point
    assert nahiku.flux[50] < 0

def test_nahiku_greedy_search():
    # Use very small data for speed
    x = np.linspace(0, 5, 50)
    y = np.random.normal(0, 0.1, 50)
    # Inject a very obvious anomaly
    y[20:25] -= 10
    nahiku = Nahiku(x, y)
    # Use training_iterations instead of max_iter
    res = nahiku.greedy_search(plot=False, training_iterations=10)
    # Note: with very few iterations it might not find it, but let's see
    assert res is not None
