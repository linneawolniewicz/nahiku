import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from nahiku.greedy_search import GreedySearch

def test_greedy_search_init():
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    gs = GreedySearch(x, y, dominant_period=2.0)
    assert gs.num_detected_anomalies == 0
    assert len(gs.flagged_anomalous) == 50

def test_greedy_search_basic():
    x = np.linspace(0, 5, 30)
    y = np.random.normal(0, 0.05, 30)
    y[10:13] -= 5.0
    gs = GreedySearch(x, y, dominant_period=1.0)
    gs.search_for_anomaly(training_iterations=5)
    assert gs.runtime > 0

def test_greedy_search_neg_only():
    x = np.linspace(0, 5, 20)
    y = np.random.normal(0, 0.05, 20)
    y[5] = 10.0 # positive spike
    y[10] = -10.0 # negative dip
    gs = GreedySearch(x, y, dominant_period=1.0)
    gs.search_for_anomaly(neg_anomaly_only=True, training_iterations=2)
    assert gs.runtime >= 0

def test_greedy_search_pos_only():
    x = np.linspace(0, 5, 20)
    y = np.random.normal(0, 0.05, 20)
    y[10] = 10.0 # positive spike
    gs = GreedySearch(x, y, dominant_period=1.0)
    gs.search_for_anomaly(pos_anomaly_only=True, training_iterations=2)
    assert gs.runtime >= 0

def test_greedy_search_plot():
    # Use a bit more data to avoid edge issues
    x = np.linspace(0, 10, 100)
    y = np.random.normal(0, 0.05, 100)
    # Give it a VERY obvious anomaly so it doesn't fail threshold checks if it looks for them
    y[40:60] -= 20.0
    gs = GreedySearch(x, y, dominant_period=1.0)
    # search_for_anomaly with plot=True
    gs.search_for_anomaly(plot=True, training_iterations=1)
    gs.plot()
    assert gs.runtime >= 0
