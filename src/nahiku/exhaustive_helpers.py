"""Retained research helpers, excluded from the installed package.

These helpers do not implement the bounded covariance-recovery policy used by
the supported searches. Use Nahiku.exhaustive_search for maintained behavior.
"""

import torch
from scipy.stats import chi2
from linear_operator.operators import DenseLinearOperator
from gpytorch.distributions import MultivariateNormal 

@torch.no_grad()
def precompute_precision(full_x, mean_module, kernel_module, noise_variance,
                         dtype=torch.float64, device="cpu"):
    """Build μ and J = (K + σ²I)^{-1} for the *full* X once.
    Use float64 here for numerical stability; you can cast later if desired.

    :param full_x: Full input grid, with shape (N, D).
    :type full_x: torch.Tensor
    :param mean_module: Callable prior mean module.
    :type mean_module: gpytorch.means.Mean
    :param kernel_module: Callable covariance module.
    :type kernel_module: gpytorch.kernels.Kernel
    :param noise_variance: Observation noise variance added to the covariance diagonal.
    :type noise_variance: float
    :param dtype: Floating-point dtype used for computation.
    :type dtype: torch.dtype
    :param device: Computation device; CPU is the default and CUDA requires an available GPU.
    :type device: str or torch.device
    :returns: Full-grid mean and observation precision matrix.
    :rtype: tuple
    """
    X = full_x.to(device=device, dtype=dtype)
    N = X.shape[0]

    # Mean on full grid
    mu = mean_module(X)            # shape [N]

    # Full noisy covariance
    # kernel_module(X) returns a LazyTensor; materialize once.
    K = kernel_module(X).to_dense()
    K = K + noise_variance * torch.eye(N, dtype=dtype, device=device)

    # Cholesky + precision
    L = torch.linalg.cholesky(K)           # O(N^3) once
    I = torch.eye(N, dtype=dtype, device=device)
    J = torch.cholesky_solve(I, L)         # (K + σ²I)^{-1}

    return mu, J

@torch.no_grad()
def interval_posterior_from_precision(mu, J, full_y, mask_train, mask_test, dtype=torch.float64):
    """Compute the posterior distribution p(y_test | y_train) using precision matrix J and mean mu.
    The training and test masks must partition the full observation grid.
    The calculation uses the corresponding blocks of the precision matrix.

    :param mu: Prior mean on the full grid, with shape (N,).
    :type mu: torch.Tensor
    :param J: Full observation precision matrix, with shape (N, N).
    :type J: torch.Tensor
    :param full_y: Observed flux on the full grid, with shape (N,).
    :type full_y: torch.Tensor
    :param mask_train: Boolean mask selecting conditioning observations.
    :type mask_train: array-like
    :param mask_test: Boolean mask selecting held-out observations.
    :type mask_test: array-like
    :param dtype: Accepted for compatibility; computations use mu.dtype.
    :type dtype: torch.dtype
    :returns: Conditional distribution over mask_test.
    :rtype: gpytorch.distributions.MultivariateNormal
    """
    device, dtype = mu.device, mu.dtype
    mask_train = torch.as_tensor(mask_train, device=device)
    mask_test  = torch.as_tensor(mask_test,  device=device)

    idx_T = mask_train.nonzero(as_tuple=True)[0]
    idx_S = mask_test.nonzero(as_tuple=True)[0]

    # Slice J into blocks; J == (K + σ²I)^{-1} == [J_SS  J_ST; J_TS  J_TT]
    if dtype == torch.float32:
        J_SS = J[idx_S][:, idx_S].float()
        J_ST = J[idx_S][:, idx_T].float()
    else:
        J_SS = J[idx_S][:, idx_S].double()
        J_ST = J[idx_S][:, idx_T].double()

    # Cov(y_S | y_T) = J_SS^{-1}
    L_SS = torch.linalg.cholesky(J_SS)
    cov_S = torch.cholesky_inverse(L_SS) # == J_SS^{-1} == Cov(y_S | y_T)

    # Mean: μ_S - J_SS^{-1} J_ST (y_T - μ_T)
    if dtype == torch.float32:
        r_T = (full_y[idx_T] - mu[idx_T]).unsqueeze(-1).float() # (y_T - μ_T)
    else:
        r_T = (full_y[idx_T] - mu[idx_T]).unsqueeze(-1).double() # (y_T - μ_T)
    rhs = J_ST @ r_T # J_ST (y_T - μ_T)
    delta = torch.cholesky_solve(rhs, L_SS).squeeze(-1) # J_SS^{-1} (J_ST (y_T - μ_T)), solved as J_SS delta = rhs; cholesky_solve(B, L) solves Ax = B where LL^T = A
    mean_S = mu[idx_S] - delta # μ_S - J_SS^{-1} J_ST (y_T - μ_T)

    return MultivariateNormal(mean_S, DenseLinearOperator(cov_S))


def compute_interval_pvalue(y_true, mvn_pred):
    """This function computes the Mahalanobis distance and corresponding p-value for a true observation under a predicted multivariate normal distribution. 
    It tries Cholesky factorization, then absolute diagonal jitter values from
    1e-6 through 1e-3, and finally a generic linear solve if factorization fails.
    The final fallback does not establish a valid positive-definite Gaussian covariance. The p-value is computed based on the chi-squared distribution with degrees of freedom 
    equal to the dimensionality of the data.

    :param y_true: Observed flux for the held-out interval.
    :type y_true: torch.Tensor
    :param mvn_pred: Conditional multivariate normal for the held-out observations.
    :type mvn_pred: gpytorch.distributions.MultivariateNormal
    :returns: Mahalanobis distance and chi-squared survival probability.
    :rtype: tuple
    """
    mean = mvn_pred.mean  # shape (d,)
    cov = mvn_pred.covariance_matrix  # shape (d, d)
    delta = y_true - mean  # shape (d,)

    alpha = None
    success = False

    # Try Cholesky decomposition directly
    try:
        L = torch.linalg.cholesky(cov)
        alpha = torch.cholesky_solve(delta.unsqueeze(-1), L)
        success = True
    except RuntimeError:
        # Log eigenvalue stats
        eigvals = torch.linalg.eigvalsh(cov)
        print(f"[Cholesky Error] Eigenvalue stats: min={eigvals.min().item():.4e}, "
              f"median={eigvals.median().item():.4e}, max={eigvals.max().item():.4e}")
        print("Attempting Cholesky with added jitter...")

        # Try adding increasing jitter to the diagonal
        for jitter_scale in [1e-6, 1e-5, 1e-4, 1e-3]:
            try:
                jitter = jitter_scale * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
                L = torch.linalg.cholesky(cov + jitter)
                alpha = torch.cholesky_solve(delta.unsqueeze(-1), L)
                success = True
                print(f"Cholesky succeeded with jitter = {jitter_scale:.0e}")
                break
            except RuntimeError:
                continue

    if not success:
        print("All Cholesky attempts failed. Falling back to direct solve (cov may not be PSD).")
        alpha = torch.linalg.solve(cov, delta.unsqueeze(-1))

    # Mahalanobis distance squared = delta.T @ cov^-1 @ delta 
    maha_sq = (delta.unsqueeze(0) @ alpha).item()
    maha_dist = maha_sq**0.5

    # p-value = P[Chi2(df) ≥ maha_sq]
    df = y_true.numel()
    p_val = chi2.sf(maha_sq, df)

    return maha_dist, p_val