"""Focused checks for bounded covariance recovery and failure reporting."""

import numpy as np
import pytest
import torch
from nahiku import Nahiku
from nahiku.numerics import stable_cholesky, CovarianceError
from nahiku.gp_helpers import condition, condition_precision
from nahiku.torch_backend import condition_torch
from nahiku.search import interval_score


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_symmetry_jitter_and_input_preservation(backend, dtype):
    original = np.ones((2, 2), dtype=dtype)
    matrix = torch.as_tensor(original.copy()) if backend == "torch" else original.copy()
    before = matrix.clone() if backend == "torch" else matrix.copy()
    records = []
    factor, adjusted = stable_cholesky(matrix, records, "test covariance")
    array = lambda a: a.numpy() if backend == "torch" else a
    np.testing.assert_array_equal(array(matrix), array(before))
    assert any(e["action"] == "jitter" for e in records)
    event = next(e for e in records if e["action"] == "jitter")
    assert 0 < event["relative_jitter"] <= 1e-6
    np.testing.assert_allclose(
        array(adjusted), original + np.eye(2) * event["absolute_jitter"], rtol=1e-6
    )
    np.testing.assert_allclose(array(factor @ factor.T), array(adjusted), rtol=1e-6)
    asymmetric = np.array([[2.0, 0.3], [0.30001, 1.0]], dtype=dtype)
    _, symmetric = stable_cholesky(
        torch.as_tensor(asymmetric) if backend == "torch" else asymmetric, records
    )
    np.testing.assert_allclose(array(symmetric), (asymmetric + asymmetric.T) / 2)
    assert records[-1]["action"] == "symmetrize"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_precision_failure_falls_back_before_jitter(backend):
    cov = np.array([[1.0, 0.2, 0.1], [0.2, 1.2, 0.3], [0.1, 0.3, 0.9]])
    mean = np.zeros(3)
    flux = np.array([0.1, 0.5, -0.2])
    test = np.array([0, 1])
    train = np.array([2])
    broken = np.eye(3)
    broken[0, 0] = -1
    records = []
    expected = condition(cov, mean, flux, train, test, diagnostics=[])
    if backend == "numpy":
        actual = condition_precision(broken, mean, flux, test, cov=cov, diagnostics=records)
    else:
        actual = condition_torch(
            torch.tensor(cov),
            torch.tensor(mean),
            torch.tensor(flux),
            train,
            test,
            torch.tensor(broken),
            records,
        )
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert any(e["action"] == "direct_fallback" for e in records)
    assert not any(e["action"] == "jitter" for e in records)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_invalid_covariance_fails_after_bounded_attempts(backend):
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    if backend == "torch":
        matrix = torch.tensor(matrix)
    with pytest.raises(CovarianceError, match="minimum eigenvalue=.*maximum numerical jitter="):
        stable_cholesky(matrix, [], context="unrecoverable interval")
    bad = np.array([[float("nan")]])
    with pytest.raises(CovarianceError, match="non-finite"):
        stable_cholesky(torch.tensor(bad) if backend == "torch" else bad, [])


def test_score_uses_adjusted_covariance_consistently():
    cov = np.ones((2, 2))
    records = []
    observed = np.array([0.1, -0.1])
    score = interval_score(observed, np.zeros(2), cov, "mll", diagnostics=records)
    jitter = next(e["absolute_jitter"] for e in records if e["action"] == "jitter")
    expected = -0.5 * np.mean(np.log(2 * np.pi * (1 + jitter)) + observed**2 / (1 + jitter))
    assert score == pytest.approx(expected)
    with pytest.raises(CovarianceError):
        interval_score(
            observed, np.zeros(2), np.array([[1.0, 2.0], [2.0, 1.0]]), "pvalue", diagnostics=[]
        )


def test_search_records_fallback_without_changing_noise(monkeypatch):
    x = np.arange(12.0)
    lc = Nahiku(x, np.sin(x), np.full(12, 0.1)).fit_gp(kernel="rbf", iterations=2)
    noise = lc._fit.noise_variance
    original_cov = lc._fit.cov.copy()
    monkeypatch.setattr(lc._fit, "_precision", -np.eye(12))
    kwargs = dict(min_points=2, max_points=2, step=4, backend="numpy", threshold=None, top_k=1)
    recovered = lc.exhaustive_search(**kwargs)
    direct = lc.exhaustive_search(**kwargs, acceleration=False)
    np.testing.assert_array_equal(recovered.mask, direct.mask)
    assert any(
        e["action"] == "direct_fallback" for e in recovered.settings["numerical_diagnostics"]
    )
    assert lc._fit.noise_variance == noise
    np.testing.assert_array_equal(lc._fit.cov, original_cov)
