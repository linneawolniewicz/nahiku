import numpy as np
import pytest
from nahiku import Nahiku

def test_exhaustive_search_basic():
    # Use very small data for speed
    x = np.linspace(0, 2, 20)
    y = np.random.normal(0, 0.001, 20)
    y[10:13] -= 10.0
    
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=4, 
        window_slide_step=1, 
        window_size_step=1,
        plot=False,
        training_iterations=5
    )
    assert res is not None

def test_exhaustive_search_pval():
    x = np.linspace(0, 2, 20)
    y = np.random.normal(0, 0.001, 20)
    y[10:13] -= 20.0
    
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=4, 
        window_slide_step=2, 
        window_size_step=2,
        which_test_metric="pval",
        plot=False,
        training_iterations=5
    )
    assert res is not None

def test_exhaustive_search_dp():
    # Test dynamic programming mode
    x = np.linspace(0, 2, 20)
    y = np.random.normal(0, 0.001, 20)
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=3, 
        dynamic_programming=True,
        refit=False,
        training_iterations=5,
        plot=False
    )
    assert res is not None

def test_exhaustive_search_refit():
    # Test refit=True
    x = np.linspace(0, 2, 10)
    y = np.random.normal(0, 0.001, 10)
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=3, 
        refit=True,
        training_iterations=2,
        plot=False
    )
    assert res is not None

def test_exhaustive_search_flag_num():
    # Test num_intervals_to_flag
    x = np.linspace(0, 2, 15)
    y = np.random.normal(0, 0.001, 15)
    nahiku = Nahiku(x, y)
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=3, 
        num_intervals_to_flag=2,
        threshold=None,
        training_iterations=2,
        plot=False
    )
    assert len(nahiku.anomalies["identified"]) > 0
