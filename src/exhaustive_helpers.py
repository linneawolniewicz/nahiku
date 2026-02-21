import torch
from scipy.stats import chi2
from gpytorch.distributions import MultivariateNormal 
from gpytorch.lazy import NonLazyTensor
from scipy.stats import chi2

@torch.no_grad()
def precompute_precision(full_x, mean_module, kernel_module, noise_variance,
                         dtype=torch.float64, device="cpu"):
    """
    Build μ and J = (K + σ²I)^{-1} for the *full* X once.
    Use float64 here for numerical stability; you can cast later if desired.

    :param full_x (torch.Tensor): (N, D) tensor of all input locations
    :param mean_module (GPyTorch mean module): GPyTorch mean module that can be called on full_x
    :param kernel_module (GPyTorch kernel module): GPyTorch kernel module that can be called on full_x to produce a LazyTensor covariance
    :param noise_variance (float): observation noise variance σ²
    :param dtype (torch.dtype): torch dtype for computations (default: torch.float64 for stability)
    :param device (str): torch device for computations (default: "cpu")
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
    """
    Compute the posterior distribution p(y_test | y_train) using precision matrix J and mean mu.
    This function computes the conditional posterior distribution over test points given training data,
    using the precision (inverse covariance) matrix and the prior mean. It leverages the block structure
    of the precision matrix to efficiently compute the conditional distribution.

    :param mu (torch.Tensor): (N,) mean vector over the full grid
    :param J (torch.Tensor): (N, N) precision matrix over the full grid, where J = (K + σ²I)^{-1}
    :param full_y (torch.Tensor): (N,) output values on the full grid
    :param mask_train (torch.Tensor or bool array): (N,) boolean mask indicating training data points
    :param mask_test (torch.Tensor or bool array): (N,) boolean mask indicating test data points
    :param dtype (torch.dtype): Data type for computations (default: torch.float64)
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

    return MultivariateNormal(mean_S, NonLazyTensor(cov_S))


def compute_interval_pvalue(y_true, mvn_pred):
    """
    This function computes the Mahalanobis distance and corresponding p-value for a true observation under a predicted multivariate normal distribution. 
    It attempts to compute the Mahalanobis distance using Cholesky decomposition for numerical stability, and falls back to direct solving 
    if Cholesky fails (e.g., if the covariance is not positive definite). The p-value is computed based on the chi-squared distribution with degrees of freedom 
    equal to the dimensionality of the data.

    :param y_true (torch.Tensor): shape (d,)
    :param mvn_pred (MultivariateNormal): MultivariateNormal from GPyTorch (mean: (d,), covar: (d,d))
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