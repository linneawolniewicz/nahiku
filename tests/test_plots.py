import numpy as np
import pytest
import matplotlib
import os
import torch

matplotlib.use("Agg")
from nahiku import Nahiku
from nahiku.balmung import Balmung
from nahiku.exhaustive_search import ExhaustiveSearch
from nahiku.greedy_search import GreedySearch


def test_nahiku_plots():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    n = Nahiku(x, y)
    # Test plot()
    n.plot(show_identified_points=True)
    # Test plot with anomalies
    n.anomalies["identified"] = [10, 11]
    n.plot(show_identified_points=True)
    # Test prewhiten plot
    # Use maxiter=1 and minimum_snr=0 to ensure something happens
    n.prewhiten(plot=True, maxiter=1, minimum_snr=0)


def test_balmung_plots():
    t = np.linspace(0, 10, 50)
    y = np.sin(t)
    bm = Balmung(t, y)
    bm.plot_lc()
    bm.plot_residual()
    # Need to run prewhiten to have self.removed for plot()
    bm.prewhiten(maxiter=1, minimum_snr=0)
    bm.plot()


def test_exhaustive_plots(tmp_path):
    x = np.linspace(0, 5, 20)
    y = np.sin(x)
    es = ExhaustiveSearch(x, y, dominant_period=1.0)
    # Run a tiny search with plot=True
    es.search_for_anomaly(
        min_anomaly_len=2, max_anomaly_len=3, plot=True, training_iterations=1
    )
    es.plot()

    # Test save_to_txt branch
    filename = tmp_path / "results.txt"
    es.search_for_anomaly(
        filename=str(filename),
        min_anomaly_len=2,
        max_anomaly_len=3,
        training_iterations=1,
    )
    assert os.path.exists(filename)


def test_greedy_plots():
    # Use standard sizes and NO search before plot_greedy
    x_np = np.linspace(0, 10, 50)
    y_np = np.sin(x_np)
    gs = GreedySearch(x_np, y_np, dominant_period=1.0)

    # Direct plot_greedy testing with matching sizes and safe indices
    # Use tensors as expected by plot_greedy
    xt = torch.tensor(x_np, dtype=torch.float32)
    pred = np.zeros(50)
    res = np.zeros(50)
    gs.plot_greedy(xt, pred, 5, 10, res)

    # Now run search and test main plot
    gs.search_for_anomaly(plot=False, training_iterations=1)
    gs.plot()


def test_balmung_extra():
    t = np.linspace(0, 10, 100)
    y = np.sin(t)
    bm = Balmung(t, y)
    bkg = bm.estimate_background(t, np.abs(y))
    assert len(bkg) == 100
