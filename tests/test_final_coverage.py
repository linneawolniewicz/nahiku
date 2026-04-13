import numpy as np
import pytest
from nahiku import Nahiku


def test_exhaustive_search_extra():
    x = np.linspace(0, 5, 20)
    y = np.random.normal(0, 0.05, 20)
    # Add a flare
    y[5:8] += 10.0
    n = Nahiku(x, y)

    # Trigger pos_anomaly_only
    n.exhaustive_search(
        min_anomaly_len=2,
        max_anomaly_len=3,
        pos_anomaly_only=True,
        training_iterations=1,
    )

    # Trigger neg_anomaly_only with a flare (should not flag)
    n.exhaustive_search(
        min_anomaly_len=2,
        max_anomaly_len=3,
        neg_anomaly_only=True,
        training_iterations=1,
    )


def test_greedy_search_extra():
    x = np.linspace(0, 5, 30)
    y = np.random.normal(0, 0.05, 30)
    y[10:15] -= 5.0
    n = Nahiku(x, y)

    n.greedy_search(update_threshold=True, training_iterations=1)
    n.greedy_search(which_grow_metric="nlpd", training_iterations=1)
    n.greedy_search(which_grow_metric="msll", training_iterations=1)
    n.greedy_search(which_grow_metric="rmse", training_iterations=1)


def test_nahiku_more_synthetic_full():
    n = Nahiku.from_synthetic_parameterized_gp(
        num_days=5, num_steps=50, add_high_residuals=True, num_high_residuals=2
    )
    # Test prewhiten with fmin/fmax
    n.prewhiten(fmin=0.1, fmax=10.0, maxiter=1, diagnose=False)

    # Test plot_dominant_period
    n.get_dominant_period(plot=True)
