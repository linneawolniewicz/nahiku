import torch
import numpy as np
import gpytorch
from nahiku.exhaustive_helpers import (
    precompute_precision,
    interval_posterior_from_precision,
)


def test_precision_and_posterior():
    N = 20
    full_x = torch.linspace(0, 1, N).unsqueeze(-1)
    full_y = torch.sin(full_x * 2 * np.pi).squeeze()

    mean_module = gpytorch.means.ConstantMean()
    kernel_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    noise_variance = 0.01

    mu, J = precompute_precision(full_x, mean_module, kernel_module, noise_variance)

    assert mu.shape == (N,)
    assert J.shape == (N, N)

    mask_train = torch.ones(N, dtype=torch.bool)
    mask_test = torch.zeros(N, dtype=torch.bool)
    mask_train[5:10] = False
    mask_test[5:10] = True

    mvn = interval_posterior_from_precision(mu, J, full_y, mask_train, mask_test)

    assert mvn.mean.shape == (5,)
    assert mvn.covariance_matrix.shape == (5, 5)
