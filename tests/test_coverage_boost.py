import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import nahiku
from nahiku import Nahiku
from nahiku.balmung import Balmung
from nahiku.gp_helpers import ParameterizedGPModel
import gpytorch
import torch

def test_hello():
    assert nahiku.hello() == "Hello from nahiku!"

def test_balmung_edge_cases():
    t = np.linspace(0, 10, 100)
    y = np.sin(t)
    bm = Balmung(t, y)
    # Trigger line 166 in balmung.py (fmax calculation)
    _, _ = bm.amplitude_spectrum(t, y, fmax=None)
    
    # Trigger find_highest_peak edge cases
    f = np.array([1, 2, 3])
    a = np.array([10, 5, 2])
    # find_highest_peak returns the frequency, not the index
    assert bm.find_highest_peak(f, a) == 1
    # imax == last
    a = np.array([2, 5, 10])
    assert bm.find_highest_peak(f, a) == 3

def test_parameterized_gp_model():
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    mean = gpytorch.means.ConstantMean()
    model = ParameterizedGPModel(kernel, mean)
    x = torch.randn(5, 1)
    res = model(x)
    assert res.mean.shape == (5,)

def test_nahiku_more_synthetic():
    # To trigger the warning safely, use a larger negative number.
    n1 = Nahiku.from_synthetic_parameterized_noise(num_steps=-1000)
    assert len(n1.time) == 1000
    
    n2 = Nahiku.from_synthetic_parameterized_noise(period=5.0, amp=0.5, phase=0.1, slope=0.001)
    assert n2.dominant_period > 0

def test_nahiku_get_dominant_period_flat():
    # Trigger the new safety branch for flat signals
    x = np.linspace(0, 10, 100)
    y = np.ones(100) # Perfectly flat
    # get_dominant_period is called in __init__
    n = Nahiku(x, y)
    assert n.dominant_period == 10.0

def test_nahiku_freq_idx_to_period_days():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    n = Nahiku(x, y)
    res = n.freq_idx_to_period_days(np.array([1.0, 2.0]), x)
    assert len(res) == 2

def test_nahiku_inject_other_shapes():
    x = np.linspace(0, 10, 100)
    y = np.random.normal(0, 0.01, 100)
    n = Nahiku(x, y)
    # Inject sawtooth and tophat
    n.inject_anomaly(1, absolute_width=0.5, absolute_depth=5, shapes=["saw"], idxs=[30])
    n.inject_anomaly(1, absolute_width=0.5, absolute_depth=5, shapes=["tophat"], idxs=[60])
    # Trigger invalid shape warning
    n.inject_anomaly(1, absolute_width=0.5, absolute_depth=5, shapes=["invalid"], idxs=[80])
    assert 30 in n.anomalies["injected"]
    assert 60 in n.anomalies["injected"]
    assert 80 in n.anomalies["injected"]
