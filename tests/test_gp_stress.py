"""GP-generated conditioning stress cases, with explicit accuracy and recovery regimes."""

from functools import lru_cache

import gpytorch
import numpy as np
import pytest
import torch

from nahiku import Nahiku
from nahiku.gp_helpers import condition, condition_precision
from nahiku.numerics import CovarianceError, stable_cholesky
from nahiku.search import interval_score
from nahiku.torch_backend import condition_torch
from test_conditioning_backends import reference


@lru_cache(None)
def kernel_matrix(kind):
    """Evaluate real GPyTorch kernels on closely spaced, uneven observations."""
    x = np.linspace(0, 0.06, 18) + 0.0001 * np.sin(np.arange(18))
    rbf = gpytorch.kernels.RBFKernel().double()
    rbf.lengthscale = 2.0
    kernel = rbf
    if kind == "quasiperiodic":
        periodic = gpytorch.kernels.PeriodicKernel().double()
        periodic.period_length = 0.7
        periodic.lengthscale = 3.0
        kernel = periodic * rbf
    with torch.no_grad():
        covariance = kernel(torch.tensor(x)).to_dense().numpy()
    return x, covariance


def calculate(cov, flux, test, dtype, backend, accelerated):
    """Use production factorization and conditioning; retain every recovery record."""
    records = []
    train = np.setdiff1d(np.arange(len(flux)), test)
    cov = cov.astype(dtype)
    flux = flux.astype(dtype)
    mean = np.zeros_like(flux)
    if backend == "torch":
        cov, flux, mean = map(torch.as_tensor, (cov, flux, mean))
    factor, effective = stable_cholesky(cov, records, "stress full GP covariance")
    if backend == "torch":
        precision = torch.cholesky_inverse(factor) if accelerated else None
        result = condition_torch(effective, mean, flux, train, test, precision, records)
        effective = effective.numpy()
    else:
        if accelerated:
            from scipy.linalg import cho_solve

            precision = cho_solve((factor, True), np.eye(len(flux), dtype=dtype))
            result = condition_precision(precision, mean, flux, test, effective, records)
        else:
            result = condition(effective, mean, flux, train, test, records)
    return result, records, effective


WINDOWS = {
    "left": np.arange(3),
    "right": np.arange(15, 18),
    "single": np.array([8]),
    "long": np.arange(4, 14),
}


@pytest.mark.parametrize("kind", ["rbf", "quasiperiodic"])
@pytest.mark.parametrize("noise_variance", [1e-2, 1e-3])
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("accelerated", [False, True])
def test_correlated_gp_accuracy(kind, noise_variance, window, dtype, backend, accelerated):
    """In the resolvable regime, all paths must agree with a float64 GPyTorch posterior."""
    x, kernel = kernel_matrix(kind)
    cov = kernel + noise_variance * np.eye(len(x))
    assert 1e3 < np.linalg.cond(cov) < 2e4
    flux = 0.2 * np.sin(x * 8) + 0.01 * np.cos(np.arange(len(x)))
    test = WINDOWS[window]
    train = np.setdiff1d(np.arange(len(x)), test)
    expected = reference(
        np.zeros(len(x)), kernel, flux, np.full(len(x), noise_variance), train, test, torch.float64
    )
    (mean, covariance), records, _ = calculate(cov, flux, test, dtype, backend, accelerated)
    # float32 budgets are below 1% of conditional uncertainty/variance, not arbitrary unit errors.
    tolerance = 0.005 if dtype == "float32" else 1e-8
    std_scale = np.sqrt(np.max(np.diag(expected[1])))
    cov_scale = np.max(np.diag(expected[1]))
    np.testing.assert_allclose(mean, expected[0], atol=tolerance * std_scale, rtol=0)
    np.testing.assert_allclose(covariance, expected[1], atol=tolerance * cov_scale, rtol=0)
    score = interval_score(flux[test], mean, covariance, "nlpd", diagnostics=records)
    density = gpytorch.distributions.MultivariateNormal(
        torch.tensor(expected[0]), torch.tensor(expected[1])
    )
    expected_score = -density.log_prob(torch.tensor(flux[test])).item() / len(test)
    assert score == pytest.approx(expected_score, abs=tolerance, rel=tolerance)
    assert not any(r["action"] in ("jitter", "direct_fallback") for r in records)


@pytest.mark.parametrize("kind", ["rbf", "quasiperiodic"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("accelerated", [False, True])
def test_low_noise_gp_float64_accuracy(kind, backend, accelerated):
    """Float64 must still resolve a GP whose condition number exceeds a billion."""
    x, kernel = kernel_matrix(kind)
    variance = 1e-8
    cov = kernel + variance * np.eye(len(x))
    assert np.linalg.cond(cov) > 1e9
    flux = 0.2 * np.sin(x * 8)
    test = WINDOWS["long"]
    train = np.setdiff1d(np.arange(len(x)), test)
    expected = reference(
        np.zeros(len(x)), kernel, flux, np.full(len(x), variance), train, test, torch.float64
    )
    (mean, covariance), records, _ = calculate(cov, flux, test, "float64", backend, accelerated)
    np.testing.assert_allclose(mean, expected[0], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(covariance, expected[1], atol=1e-12, rtol=1e-4)
    assert not any(r["action"] == "jitter" for r in records)


@pytest.mark.parametrize("kind", ["rbf", "quasiperiodic"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("accelerated", [False, True])
def test_below_float32_resolution_reports_recovery(kind, backend, accelerated):
    """Lost noise precision must be visible; successful recovery is not an accuracy certificate."""
    x, kernel = kernel_matrix(kind)
    cov = kernel + 1e-12 * np.eye(len(x))
    rounded = cov.astype("float32")
    # The observation noise is genuinely lost, without manually corrupting the precision cache.
    np.testing.assert_array_equal(np.diag(rounded), np.diag(kernel.astype("float32")))
    flux = 0.2 * np.sin(x * 8)
    test = WINDOWS["long"]
    (mean, covariance), records, effective = calculate(
        cov, flux, test, "float32", backend, accelerated
    )
    jitters = [r for r in records if r["action"] == "jitter"]
    assert jitters, "GP below float32 noise resolution must report its regularization"
    assert all(0 < r["relative_jitter"] <= 1e-6 for r in jitters)
    assert np.isfinite(mean).all() and np.isfinite(covariance).all()
    np.linalg.cholesky(covariance.astype("float64"))
    assert np.max(np.abs(effective - rounded)) <= 2e-6
    # Recovered variances must not be represented as resolving the original 1e-12 noise.
    assert np.max(np.diag(covariance)) > 1e-10


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_fitted_searches_recover_known_dip(dtype, backend):
    """Both search algorithms retain a well-separated detection in a correlated low-noise fit."""
    x = np.linspace(0, 1, 32)
    flux = 0.1 * np.sin(x * 2)
    flux[14:17] -= 0.4
    errors = np.full(len(x), 0.003)
    lc = Nahiku(x, flux, errors).fit_gp(
        kernel="rbf",
        length_scale=3,
        noise=0.003,
        iterations=2,
        learning_rate=1e-5,
        standardize=False,
        dtype=dtype,
    )
    assert np.linalg.cond(lc._fit.cov.astype("float64")) > 1e3
    cov_before = lc._fit.cov.copy()
    noise_before = lc._fit.noise_variance
    for accelerated in (False, True):
        result = lc.exhaustive_search(
            min_points=3,
            max_points=3,
            step=1,
            metric="nlpd",
            threshold=None,
            top_k=1,
            backend=backend,
            acceleration=accelerated,
        )
        assert result.intervals == [(14, 17)]
    greedy = lc.greedy_search(
        seed_statistic="loo",
        growth="threshold",
        sigma=4,
        grow_sigma=2,
        direction="negative",
        max_events=1,
        backend=backend,
    )
    assert greedy.mask[14:17].all()
    assert not greedy.mask[:10].any() and not greedy.mask[22:].any()
    assert np.isfinite(greedy.std).all()
    np.testing.assert_array_equal(lc._fit.cov, cov_before)
    assert lc._fit.noise_variance == noise_before


@pytest.mark.parametrize("kind", ["rbf", "quasiperiodic"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_unresolved_near_coincident_gp_fails_without_jitter(kind, dtype, backend):
    """A real GP whose distinct times round to identical covariance rows cannot bypass recovery."""
    x = torch.linspace(0, 1e-12, 18, dtype=torch.float64)
    kernel = gpytorch.kernels.RBFKernel().double()
    kernel.lengthscale = 2.0
    if kind == "quasiperiodic":
        periodic = gpytorch.kernels.PeriodicKernel().double()
        periodic.period_length = 0.7
        kernel = kernel * periodic
    with torch.no_grad():
        matrix = (kernel(x).to_dense().numpy() + 1e-20 * np.eye(18)).astype(dtype)
    # Distinct timestamps and positive nominal noise, but insufficient representable variation.
    assert np.diff(x.numpy()).min() > 0
    np.testing.assert_array_equal(matrix, np.ones_like(matrix))
    value = torch.tensor(matrix) if backend == "torch" else matrix
    with pytest.raises(CovarianceError, match="maximum numerical jitter=0"):
        stable_cholesky(value, [], "near-coincident GP without regularization", allow_jitter=False)
    records = []
    _, recovered = stable_cholesky(value, records, "near-coincident GP with regularization")
    assert any(r["action"] == "jitter" for r in records)
    recovered = recovered.numpy() if backend == "torch" else recovered
    np.linalg.cholesky(recovered.astype("float64"))
