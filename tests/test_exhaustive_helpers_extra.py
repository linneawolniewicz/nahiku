import torch
import numpy as np
from gpytorch.distributions import MultivariateNormal as GPyTorchMVN
from torch.distributions import MultivariateNormal as TorchMVN
from nahiku.exhaustive_helpers import compute_interval_pvalue, interval_posterior_from_precision

def test_compute_pvalue_cholesky_fail():
    # Create a non-positive definite covariance to trigger fallback
    mean = torch.zeros(2)
    # This matrix is singular/not PD
    cov = torch.tensor([[1.0, 2.0], [2.0, 4.0]])
    
    class DummyMVN:
        def __init__(self, loc, covariance_matrix):
            self.mean = loc
            self.covariance_matrix = covariance_matrix
    
    mvn = DummyMVN(mean, cov)
    y_true = torch.tensor([1.0, 1.0])
    
    # This should trigger the try-except block in compute_interval_pvalue
    pval = compute_interval_pvalue(y_true, mvn)
    # If it returns a tuple, we check if the first element is what we want or if we should unpack it.
    # Looking at the error log: isinstance((457.94701522119345, 0.0), float) is False.
    # So it is returning a tuple (distance, p-value).
    if isinstance(pval, tuple):
        pval = pval[1]
    assert isinstance(pval, float)
    assert 0 <= pval <= 1

def test_interval_posterior_float32():
    N = 10
    mu = torch.zeros(N)
    J = torch.eye(N)
    full_y = torch.ones(N)
    mask_train = torch.ones(N, dtype=torch.bool)
    mask_test = torch.zeros(N, dtype=torch.bool)
    mask_train[0:2] = False
    mask_test[0:2] = True
    
    # Test float32 branch
    mvn = interval_posterior_from_precision(mu, J, full_y, mask_train, mask_test, dtype=torch.float32)
    assert mvn.mean.dtype == torch.float32
