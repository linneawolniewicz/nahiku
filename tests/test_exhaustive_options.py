"""Reference calculations for exhaustive metrics and conditioning backends."""

import numpy as np
import pytest
import torch
from scipy.stats import norm, multivariate_normal
from nahiku import Nahiku
from nahiku.search import interval_score
from nahiku.torch_backend import condition_torch
from test_conditioning_backends import reference


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-9), (torch.float32, 3e-5)])
@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("held", [[0], [2, 3], [0, 4, 8]])
def test_torch_interval_conditioning_against_gpytorch(dtype, tol, accelerated, held):
    rng = np.random.default_rng(19)
    a = rng.normal(size=(9, 9))
    kernel = a @ a.T / 9 + np.eye(9) * 0.2
    noise = np.linspace(0.05, 0.2, 9)
    mean = rng.normal(size=9)
    flux = rng.normal(size=9)
    test = np.array(held)
    train = np.setdiff1d(np.arange(9), test)
    tensor = lambda x: torch.as_tensor(x, dtype=dtype)
    cov = tensor(kernel + np.diag(noise))
    precision = torch.cholesky_inverse(torch.linalg.cholesky(cov)) if accelerated else None
    actual = condition_torch(cov, tensor(mean), tensor(flux), train, test, precision)
    expected = reference(mean, kernel, flux, noise, train, test, dtype)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, atol=tol, rtol=tol)


def test_log_scores_against_independent_density_formulas():
    observed = np.array([-0.7, 0.4])
    mean = np.array([0.1, -0.1])
    covariance = np.array([[0.8, 0.25], [0.25, 0.6]])
    training = np.array([-0.4, 0.1, 0.2, 0.8])
    joint = multivariate_normal.logpdf(observed, mean=mean, cov=covariance)
    marginal = norm.logpdf(observed, loc=mean, scale=np.sqrt(np.diag(covariance))).mean()
    baseline = norm.logpdf(observed, loc=training.mean(), scale=training.std()).mean()
    assert interval_score(observed, mean, covariance, "ll") == pytest.approx(joint)
    assert interval_score(observed, mean, covariance, "mll") == pytest.approx(marginal)
    assert interval_score(observed, mean, covariance, "nlpd") == pytest.approx(-joint / 2)
    assert interval_score(observed, mean, covariance, "msll", training) == pytest.approx(
        baseline - marginal
    )
    with pytest.raises(ValueError, match="reference_flux"):
        interval_score(observed, mean, covariance, "msll")


def fitted(dtype="float64"):
    x = np.arange(18.0) / 3
    y = 0.2 * np.sin(x)
    y[7:9] -= 1
    return Nahiku(x, y, np.linspace(0.08, 0.15, 18)).fit_gp(
        kernel="rbf", iterations=3, noise=0.15, dtype=dtype
    )


@pytest.mark.parametrize("metric", ["pvalue", "mahalanobis", "nlpd", "rmse", "msll", "mll", "ll"])
@pytest.mark.parametrize("dtype,tol", [("float64", 1e-8), ("float32", 2e-3)])
def test_exhaustive_backend_and_acceleration_agreement(metric, dtype, tol):
    lc = fitted(dtype)
    options = dict(
        min_points=2,
        max_points=4,
        step=4,
        size_step=2,
        threshold=None,
        top_k=2,
        direction="both",
        metric=metric,
    )
    outputs = [
        lc.exhaustive_search(**options, backend=backend, acceleration=acc)
        for backend in ("numpy", "torch")
        for acc in (False, True)
    ]
    for result in outputs[1:]:
        assert result.intervals == outputs[0].intervals
        np.testing.assert_allclose(
            [c["score"] for c in result.candidates],
            [c["score"] for c in outputs[0].candidates],
            atol=tol,
            rtol=tol,
        )
        np.testing.assert_allclose(result.mean, outputs[0].mean, atol=tol, rtol=tol)
        np.testing.assert_allclose(result.std, outputs[0].std, atol=tol, rtol=tol)


@pytest.mark.parametrize("metric,smaller", [("ll", True), ("mll", True), ("msll", False)])
def test_metric_selection_direction(metric, smaller):
    lc = fitted()
    options = dict(min_points=2, max_points=2, step=3, direction="both", metric=metric)
    ranked = lc.exhaustive_search(**options, threshold=None, top_k=1)
    scores = np.array([c["score"] for c in ranked.candidates])
    strongest = ranked.candidates[np.argmin(scores) if smaller else np.argmax(scores)]
    expected = np.zeros(18, bool)
    expected[strongest["start"] : strongest["stop"]] = True
    np.testing.assert_array_equal(ranked.mask, expected)
    cutoff = float(np.median(scores))
    selected = lc.exhaustive_search(**options, threshold=cutoff)
    expected[:] = False
    for c in selected.candidates:
        if c["score"] <= cutoff if smaller else c["score"] >= cutoff:
            expected[c["start"] : c["stop"]] = True
    np.testing.assert_array_equal(selected.mask, expected)


@pytest.mark.parametrize("warm", [False, True])
def test_exhaustive_refits_use_independent_initialization(monkeypatch, warm):
    lc = fitted()
    original = lc._refit_subset
    seen = []

    def record(indices, initial_fit=None):
        seen.append(initial_fit)
        return original(indices, initial_fit=initial_fit)

    monkeypatch.setattr(lc, "_refit_subset", record)
    options = dict(
        min_points=2,
        max_points=3,
        step=8,
        refit=True,
        acceleration=False,
        warm_initialization=warm,
        metric="mll",
        threshold=None,
        top_k=1,
    )
    a = lc.exhaustive_search(**options, backend="numpy")
    b = lc.exhaustive_search(**options)
    assert seen and all(f is (lc._fit if warm else None) for f in seen)
    np.testing.assert_allclose(
        [c["score"] for c in a.candidates], [c["score"] for c in b.candidates], atol=1e-8
    )
    np.testing.assert_array_equal(a.mask, b.mask)
    assert b.settings["backend"] == "torch"
    assert b.settings["warm_initialization"] is warm


def test_exhaustive_option_validation():
    lc = fitted()
    with pytest.raises(ValueError, match="backend"):
        lc.exhaustive_search(backend="invalid")
    with pytest.raises(ValueError, match="warm_initialization"):
        lc.exhaustive_search(warm_initialization="yes")
