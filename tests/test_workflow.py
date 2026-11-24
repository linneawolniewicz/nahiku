"""Numerical and scientific regression checks against actual dependencies."""

import json
import numpy as np
import pytest
from nahiku import Nahiku
from nahiku.gp_helpers import condition, condition_precision
from nahiku.search import evaluate_events
from nahiku.results import events_from_mask


@pytest.fixture
def fitted():
    lc = Nahiku.from_synthetic_noise(n_points=100, red_noise=0, white_noise=0.02)
    lc.inject_anomaly(center=5.0, width=0.13, depth=0.8)
    lc.standardize().fit_gp(period=2.0, length_scale=5.0, iterations=35)
    return lc


def test_constructor_preserves_flux_and_validates():
    x, y = np.arange(10.0), np.arange(10.0) + 100
    lc = Nahiku(x, y, np.ones(10))
    np.testing.assert_array_equal(lc.flux, y)
    assert lc.period is None and lc.results == {}
    with pytest.raises(ValueError):
        lc.flux[0] = 0
    with pytest.raises(ValueError):
        Nahiku([0, 1, 1], [1, 2, 3])
    with pytest.raises(ValueError):
        Nahiku([0, 1, 2], [1, np.nan, 3])
    with pytest.raises(ValueError):
        Nahiku([0, 1, 2], [1, 2, 3], [1, 0, 1])


def test_standardization_and_invalidation(fitted):
    original = fitted.original_flux.copy()
    result = fitted.greedy_search()
    old = result.flux.copy()
    fitted.standardize()
    assert fitted.results == {}
    with pytest.raises(RuntimeError):
        fitted.predict()
    np.testing.assert_array_equal(result.flux, old)
    np.testing.assert_array_equal(fitted.original_flux, original)
    fitted.reset()
    np.testing.assert_array_equal(fitted.flux, original)
    assert not fitted.truth_mask.any()
    lc = Nahiku(np.arange(5.0), np.arange(5.0), np.ones(5) * 2)
    lc.standardize()
    np.testing.assert_allclose(lc.flux_err, 2 / np.std(np.arange(5.0)))


def test_irregular_period_recovery():
    rng = np.random.default_rng(8)
    x = np.sort(rng.uniform(0, 30, 500))
    y = np.sin(2 * np.pi * x / 2.35) + rng.normal(0, 0.03, len(x))
    lc = Nahiku(x, y)
    period = lc.estimate_period(minimum_period=1, maximum_period=5)
    assert period == pytest.approx(2.35, rel=0.01)
    np.testing.assert_array_equal(lc.flux, y)
    assert Nahiku(x, np.ones(len(x))).estimate_period() is None


def test_prewhitening_recovery_and_dip_preservation():
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 10, 1000))
    y = 0.5 * np.cos(2 * np.pi * 7.3 * x + 0.4) + rng.normal(0, 0.005, len(x))
    lc = Nahiku(x, y)
    signal = lc.inject_anomaly(center=5, width=0.1, depth=0.15)
    # Independent exclusion illustrates how to protect a known event.
    result = lc.prewhiten(method="weighted", fmin=5, fmax=10, max_components=1, fit_mask=np.abs(x - 5) > 0.5)
    assert result.frequencies[0] == pytest.approx(7.3, abs=0.003)
    assert result.amplitudes[0] == pytest.approx(0.5, abs=0.01)
    assert np.std(result.residuals - signal) < 0.01
    np.testing.assert_allclose(result.model + result.residuals, y + signal)
    assert result.stopping_reason == "component_limit"


def test_conditional_precision_matches_direct():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(12, 12))
    cov = a @ a.T + np.eye(12) * 0.2
    mean, y = rng.normal(size=(2, 12))
    test = np.arange(3, 7)
    train = np.setdiff1d(np.arange(12), test)
    direct = condition(cov, mean, y, train, test)
    fast = condition_precision(np.linalg.inv(cov), mean, y, test)
    np.testing.assert_allclose(fast[0], direct[0], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fast[1], direct[1], rtol=1e-10, atol=1e-10)


def test_gp_predictions_against_dense_reference(fitted):
    fit = fitted._fit
    mean, std = fitted.predict()
    expected = fit.prior_mean + fit.kernel_cov @ np.linalg.solve(
        fit.cov, fitted.flux - fit.prior_mean
    )
    covariance = fit.kernel_cov - fit.kernel_cov @ np.linalg.solve(fit.cov, fit.kernel_cov)
    np.testing.assert_allclose(mean, expected, atol=1e-8)
    np.testing.assert_allclose(std**2, np.diag(covariance), atol=1e-8)
    _, obs_std = fitted.predict(include_noise=True)
    np.testing.assert_allclose(
        obs_std**2 - std**2, fit.noise_variance + fitted.flux_err**2, atol=1e-8
    )


def test_searches_independent_and_recover_dip(fitted):
    greedy = fitted.greedy_search(sigma=3.5, grow_sigma=1.5)
    before = greedy.mask.copy()
    exhaustive = fitted.exhaustive_search(min_points=2, max_points=8, step=3, threshold=1e-4)
    np.testing.assert_array_equal(greedy.mask, before)
    assert fitted.results["greedy"] is greedy
    assert fitted.results["exhaustive"] is exhaustive
    assert fitted.evaluate(greedy)["recall"] == 1
    assert fitted.evaluate(exhaustive)["recall"] == 1
    fast_scores = [c["score"] for c in exhaustive.candidates]
    direct = fitted.exhaustive_search(
        min_points=2, max_points=8, step=3, threshold=1e-4, acceleration=False
    )
    np.testing.assert_allclose(
        fast_scores, [c["score"] for c in direct.candidates], atol=1e-8, rtol=1e-6
    )
    assert any(c["stop"] == len(fitted.time) for c in direct.candidates)
    assert any(c["stop"] - c["start"] == 8 for c in direct.candidates)


def test_empty_and_invalid_searches(fitted):
    empty = fitted.greedy_search(sigma=1e8)
    assert empty.intervals == []
    assert np.isfinite(empty.std).all()
    with pytest.raises(ValueError):
        fitted.exhaustive_search(top_k=3)
    with pytest.raises(ValueError):
        fitted.exhaustive_search(refit=True)
    with pytest.raises(ValueError):
        fitted.greedy_search(direction="dips")
    top = fitted.exhaustive_search(min_points=2, max_points=6, step=4, threshold=None, top_k=2)
    assert len(top.intervals) <= 2


def test_event_matching_is_one_to_one():
    report = evaluate_events([(0, 20)], [(2, 4), (10, 12)])
    assert report["precision"] == 1 and report["recall"] == 0.5
    assert events_from_mask([True, True, False, True]) == [(0, 2), (3, 4)]
    assert evaluate_events([], [])["precision"] == 0


@pytest.mark.parametrize("shape", ["gaussian", "saw", "exocomet"])
def test_injections_and_reproducibility(shape):
    a = Nahiku.from_synthetic_noise(n_points=80, seed=11)
    b = Nahiku.from_synthetic_noise(n_points=80, seed=11)
    np.testing.assert_array_equal(a.flux, b.flux)
    signal = a.inject_anomaly(center=5, width=0.2, depth=0.5, shape=shape)
    assert signal.min() < -0.1 and signal.max() <= 0 and a.truth_mask.any()
    a = Nahiku.from_synthetic_gp(n_points=40, seed=11)
    b = Nahiku.from_synthetic_gp(n_points=40, seed=11)
    np.testing.assert_array_equal(a.flux, b.flux)


def test_export_and_plot(fitted, tmp_path):
    import matplotlib.pyplot as plt

    result = fitted.greedy_search(sigma=3)
    paths = result.export(tmp_path / "fit")
    assert all(p.exists() for p in paths.values())
    saved = np.load(paths["arrays"])
    np.testing.assert_array_equal(saved["mask"], result.mask)
    assert json.loads(paths["metadata"].read_text())["method"] == "greedy"
    fig, axes = result.plot()
    assert len(axes) == 2
    plt.close(fig)


def test_gap_windows_do_not_cross():
    x = np.r_[np.linspace(0, 1, 15), np.linspace(4, 5, 15)]
    lc = Nahiku(x, np.sin(x)).fit_gp(kernel="rbf", iterations=5)
    result = lc.exhaustive_search(min_points=2, max_points=5, step=1)
    assert not any(c["start"] < 15 < c["stop"] for c in result.candidates)


def test_lightkurve_conversion_real_dependency():
    lk = pytest.importorskip("lightkurve")
    lc = lk.LightCurve(time=[2.0, 0.0, 1.0, 3.0], flux=[2.0, 1.0, np.nan, 3.0], flux_err=[0.1] * 4)
    result = Nahiku.from_lightkurve(lc)
    np.testing.assert_array_equal(result.time, [0, 2, 3])
    assert result.metadata["source_indices"] == [1, 0, 3]
    np.testing.assert_allclose(result.flux_err, 0.05)


def test_bin_propagates_uncertainty_and_keeps_parent():
    lc = Nahiku(np.arange(12.0) * 0.1, np.arange(12.0), np.ones(12) * 2)
    before = lc.flux.copy()
    binned = lc.bin(0.25)
    for i, group in enumerate(binned.metadata["bin_source_groups"]):
        assert binned.flux[i] == pytest.approx(np.mean(before[group]))
        assert binned.flux_err[i] == pytest.approx(2 / np.sqrt(len(group)))
    np.testing.assert_array_equal(lc.flux, before)


def test_refit_and_alternative_metrics():
    lc = Nahiku.from_synthetic_noise(n_points=25, red_noise=0, white_noise=0.01)
    lc.inject_anomaly(center=5, width=0.3, depth=1)
    lc.standardize().fit_gp(period=2, iterations=5)
    for metric in ("mahalanobis", "nlpd", "rmse"):
        result = lc.exhaustive_search(
            min_points=2,
            max_points=3,
            step=12,
            metric=metric,
            threshold=None,
            top_k=1,
            refit=True,
            acceleration=False,
        )
        assert all(np.isfinite(c["score"]) for c in result.candidates)
    assert np.isfinite(lc.greedy_search(sigma=2, grow_sigma=1, max_events=2, refit=True).mean).all()


def test_noise_only_conservative_threshold():
    rng = np.random.default_rng(91)
    lc = Nahiku(np.arange(80.0) / 10, rng.normal(0, 0.1, 80), np.full(80, 0.1))
    lc.fit_gp(kernel="rbf", iterations=20)
    assert not lc.greedy_search(sigma=8, grow_sigma=4).mask.any()
    assert not lc.exhaustive_search(min_points=2, max_points=5, step=4, threshold=1e-12).mask.any()




def test_lightkurve_without_errors():
    lk = pytest.importorskip("lightkurve")
    lc = Nahiku.from_lightkurve(lk.LightCurve(time=[0.0, 1.0, 2.0], flux=[1.0, 2.0, 3.0]))
    assert lc.flux_err is None and len(lc.time) == 3
