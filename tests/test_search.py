import torch
import numpy as np
import gpytorch
import pytest
from nahiku.search import Search


# Search is an abstract class with abstractmethod search_for_anomaly
# We need a concrete implementation to test the base class methods
class ConcreteSearch(Search):
    def search_for_anomaly(self, **kwargs):
        pass


def test_search_init():
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    s = ConcreteSearch(x, y, dominant_period=2.0)
    assert s.dominant_period == 2.0
    assert isinstance(s.x_tensor, torch.Tensor)
    assert s.x_tensor.shape == (50,)


def test_search_build_gp_model():
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    s = ConcreteSearch(x, y, dominant_period=2.0)
    model, likelihood, kernel, mean = s.build_gp_model()
    assert isinstance(model, gpytorch.models.ExactGP)
    assert isinstance(likelihood, gpytorch.likelihoods.GaussianLikelihood)


def test_search_train_gp():
    x = np.linspace(0, 5, 20)
    y = np.sin(x)
    s = ConcreteSearch(x, y, dominant_period=1.0)
    model, likelihood, kernel, mean = s.build_gp_model()
    tx = torch.tensor(x).float().unsqueeze(-1)
    ty = torch.tensor(y).float()
    # Test a very short training run
    model, likelihood, mll = s.train_gp(
        model, likelihood, x=tx, y=ty, training_iterations=2, early_stopping=False
    )
    assert mll is not None

    # Test early stopping branch
    s.train_gp(
        model, likelihood, x=tx, y=ty, training_iterations=10, early_stopping=True
    )


def test_search_constraints():
    # Test different dominant periods to trigger different constraint logic in __init__
    s1 = ConcreteSearch(np.arange(10), np.arange(10), dominant_period=0.2)
    s2 = ConcreteSearch(np.arange(10), np.arange(10), dominant_period=0.7)
    s3 = ConcreteSearch(np.arange(10), np.arange(10), dominant_period=2.0)
    s4 = ConcreteSearch(np.arange(10), np.arange(10), dominant_period=6.0)
    assert s1.dominant_period == 0.2
    assert s2.dominant_period == 0.7
    assert s3.dominant_period == 2.0
    assert s4.dominant_period == 6.0
