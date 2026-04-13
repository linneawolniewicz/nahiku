import numpy as np
from nahiku.balmung import Balmung


def test_balmung_init():
    t = np.linspace(0, 10, 100)
    y = np.sin(t)
    b = Balmung(t, y)
    assert len(b.time) == 100
    assert len(b.flux) == 100


def test_balmung_prewhiten():
    t = np.linspace(0, 10, 500)
    # Signal with frequency 1.0, amplitude 1.0
    y = 1.0 * np.cos(2 * np.pi * 1.0 * t + 0.5)
    b = Balmung(t, y)
    b.prewhiten(maxiter=1, minimum_snr=0)
    assert len(b.removed) == 1
    # Check if the frequency is approximately correct
    # b.removed[0] is [freq, amp, phase]
    assert np.isclose(b.removed[0][0], 1.0, atol=0.1)


def test_balmung_model():
    t = np.array([0, 1, 2])
    b = Balmung(t, t)
    model_y = b.model(t, 1.0, 1.0, 0.0)
    assert len(model_y) == 3
    assert np.allclose(model_y, np.cos(2 * np.pi * 1.0 * t))


def test_balmung_negative_amp():
    t = np.linspace(0, 10, 100)
    y = np.sin(t)
    bm = Balmung(t, y)
    # Use an initial guess with negative amplitude to trigger branch in fit()
    # theta = [freq, amp, phase]
    res = bm.fit([1.0 / (2 * np.pi), -1.0, 0.0])
    assert res[1] > 0
