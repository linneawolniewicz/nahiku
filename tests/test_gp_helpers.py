import torch
import gpytorch
import numpy as np
from nahiku.gp_helpers import QuasiPeriodicKernel, ExactGPModel


def test_quasi_periodic_kernel():
    kernel = QuasiPeriodicKernel()
    x1 = torch.randn(10, 1)
    x2 = torch.randn(10, 1)
    res = kernel(x1, x2)
    assert res.shape == (10, 10)


def test_exact_gp_model():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.sin(train_x * (2 * np.pi))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    mean = gpytorch.means.ConstantMean()
    model = ExactGPModel(train_x, train_y, likelihood, kernel, mean)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 5)
        observed_pred = likelihood(model(test_x))
        assert observed_pred.mean.shape == (5,)
