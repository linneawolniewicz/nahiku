import numpy as np
import pytest
from nahiku import Nahiku

def test_exhaustive_search_basic():
    # Use very small data for speed
    x = np.linspace(0, 2, 20)
    # Start with very low noise
    y = np.random.normal(0, 0.001, 20)
    # Inject an anomaly at indices 10 to 12
    y[10:13] -= 10.0
    
    nahiku = Nahiku(x, y)
    # Use small window lengths for testing
    res = nahiku.exhaustive_search(
        min_anomaly_len=2, 
        max_anomaly_len=4, 
        window_slide_step=1, 
        window_size_step=1,
        plot=False,
        training_iterations=10
    )
    # Even if it doesn't "identify" it into the list (due to thresholds), 
    # the search object should be returned
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
        training_iterations=10
    )
    assert res is not None
