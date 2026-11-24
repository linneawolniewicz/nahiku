"""Scientific and behavioral checks for configurable greedy searches."""

import numpy as np
import pytest
from nahiku import Nahiku
from nahiku.greedy_search import _growth_loss


def lightcurve(flux=None, time=None):
    if flux is None:
        flux = np.zeros(24)
        flux[4:6] = -1
        flux[16:18] = 0.8
    if time is None:
        time = np.arange(len(flux), dtype=float)
    return Nahiku(time, flux, np.full(len(flux), 0.05)).fit_gp(
        kernel="rbf",
        length_scale=100,
        noise=0.3,
        iterations=1,
        learning_rate=1e-12,
        standardize=False,
    )


@pytest.mark.parametrize("statistic", ["residual", "loo"])
@pytest.mark.parametrize(
    "direction,expected", [("negative", [4, 5]), ("positive", [16, 17]), ("both", [4, 5, 16, 17])]
)
def test_seed_sign_run_and_unlimited_stopping(statistic, direction, expected):
    lc = lightcurve()
    result = lc.greedy_search(
        seed_statistic=statistic,
        growth="threshold",
        len_deviant=2,
        direction=direction,
        sigma=2,
        grow_sigma=1,
    )
    np.testing.assert_array_equal(np.flatnonzero(result.mask), expected)
    assert result.settings["stop_reason"] == "threshold"
    assert result.settings["max_events"] is None
    assert result.settings["backend"] == "torch"
    limited = lc.greedy_search(
        seed_statistic=statistic,
        growth="threshold",
        len_deviant=2,
        sigma=2,
        grow_sigma=1,
        max_events=1,
    )
    assert len(limited.candidates) == 1
    assert limited.settings["stop_reason"] == "max_events"


@pytest.mark.parametrize("statistic", ["residual", "loo"])
def test_isolated_outlier_rejected_by_run_length(statistic):
    y = np.zeros(24)
    y[9] = -1
    lc = lightcurve(y)
    assert lc.greedy_search(
        seed_statistic=statistic, len_deviant=1, growth="threshold", sigma=2
    ).mask.any()
    assert not lc.greedy_search(
        seed_statistic=statistic, len_deviant=2, growth="threshold", sigma=2
    ).mask.any()


def test_gap_barrier_can_be_disabled_for_seed_and_result():
    y = np.zeros(24)
    y[11:13] = -1
    x = np.arange(24, dtype=float)
    x[12:] += 100
    lc = lightcurve(y, x)
    joined = lc.greedy_search(growth="threshold", sigma=2, len_deviant=2)
    assert joined.intervals == [(11, 13)]
    split = lc.greedy_search(growth="threshold", sigma=2, len_deviant=2, gap_factor=5)
    assert not split.mask.any()


def test_threshold_update_history_and_empty_detection():
    lc = lightcurve()
    fixed = lc.greedy_search(growth="threshold", sigma=2, grow_sigma=1)
    adaptive = lc.greedy_search(growth="threshold", sigma=2, grow_sigma=1, update_threshold=True)
    assert len(fixed.settings["residual_variance_history"]) >= 2
    assert np.ptp(fixed.settings["residual_variance_history"]) == 0
    assert np.ptp(adaptive.settings["residual_variance_history"]) > 0
    assert not lc.greedy_search(sigma=100).mask.any()
    with pytest.raises(ValueError, match="update_threshold"):
        lc.greedy_search(seed_statistic="loo", update_threshold=True)


@pytest.mark.parametrize("metric", ["mll", "nlpd", "msll", "rmse"])
def test_metric_growth_backends_and_valid_exclusions(metric):
    lc = lightcurve()
    kwargs = dict(growth_metric=metric, sigma=2, max_events=2)
    a = lc.greedy_search(backend="numpy", **kwargs)
    b = lc.greedy_search(backend="torch", **kwargs)
    assert a.mask.any()
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-9)
    assert (~a.mask).sum() >= 3
    assert all(0 <= c["start"] < c["stop"] <= len(lc.time) for c in a.candidates)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_mll_and_nlpd_against_gpytorch(backend):
    import torch
    import gpytorch
    from test_conditioning_backends import reference

    lc = lightcurve()
    fit = lc._fit
    retained = np.ones(len(lc.time), bool)
    retained[4:6] = False
    mean, std = fit.predict(lc.time, retained, include_noise=True, errors=lc.flux_err)
    idx = np.flatnonzero(retained)
    covariance = fit.cov[np.ix_(idx, idx)]
    prior = torch.as_tensor(fit.prior_mean[idx], dtype=fit.torch_dtype)
    observed = torch.as_tensor(lc.flux[idx], dtype=fit.torch_dtype)
    expected = -gpytorch.distributions.MultivariateNormal(
        prior, torch.as_tensor(covariance)
    ).log_prob(observed).item() / len(idx)
    assert _growth_loss(lc, fit, retained, mean, std, "mll", backend) == pytest.approx(
        expected, abs=1e-9
    )
    m, cov = reference(
        fit.prior_mean,
        fit.kernel_cov,
        fit.flux,
        fit.noise_variance + fit.errors**2,
        idx,
        idx,
        fit.torch_dtype,
    )
    expected = -gpytorch.distributions.MultivariateNormal(
        torch.as_tensor(m), torch.as_tensor(cov)
    ).log_prob(observed).item() / len(idx)
    assert _growth_loss(lc, fit, retained, mean, std, "nlpd", backend) == pytest.approx(
        expected, abs=1e-9
    )


def test_warm_start_initial_loss_and_fixed_errors():
    lc = lightcurve()
    base = lc._fit
    params = {name: p.detach().clone() for name, p in base.model.named_parameters()}
    idx = np.r_[0:4, 6:24]
    warm = lc._refit_subset(idx, initial_fit=base)
    residual = lc.flux[idx] - base.prior_mean[idx]
    cov = base.cov[np.ix_(idx, idx)]
    expected = (
        0.5
        * (
            residual @ np.linalg.solve(cov, residual)
            + np.linalg.slogdet(cov)[1]
            + len(idx) * np.log(2 * np.pi)
        )
        / len(idx)
    )
    assert warm.losses[0] == pytest.approx(expected, abs=1e-8)
    np.testing.assert_array_equal(warm.errors, lc.flux_err[idx])
    for name, parameter in base.model.named_parameters():
        np.testing.assert_array_equal(parameter.detach().numpy(), params[name].numpy())


@pytest.mark.parametrize("warm", [False, True])
def test_refit_initialization_selection(monkeypatch, warm):
    lc = lightcurve()
    original = lc._refit_subset
    previous = []

    def record(indices, initial_fit=None):
        previous.append(initial_fit)
        return original(indices, initial_fit=initial_fit)

    monkeypatch.setattr(lc, "_refit_subset", record)
    result = lc.greedy_search(refit=True, warm_initialization=warm, sigma=2, max_events=1)
    assert result.mask.any() and previous
    assert all((fit is not None) == warm for fit in previous)
    assert not warm or previous[0] is lc._fit


@pytest.mark.parametrize(
    "options",
    [
        {"max_events": 0},
        {"len_deviant": 0},
        {"expansion_param": 0},
        {"gap_factor": 0},
        {"min_remaining": 1},
        {"sigma": float("nan")},
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        lightcurve().greedy_search(**options)


def test_metric_growth_stops_at_last_improving_interval():
    from scipy.stats import multivariate_normal

    lc = lightcurve()
    fit = lc._fit
    a, b = 4, 6
    losses = []
    intervals = []
    # Independently evaluate all symmetric exclusions with SciPy's Gaussian density.
    while len(lc.time) - (b - a) >= 3:
        keep = np.ones(len(lc.time), bool)
        keep[a:b] = False
        losses.append(
            -multivariate_normal.logpdf(
                lc.flux[keep], mean=fit.prior_mean[keep], cov=fit.cov[np.ix_(keep, keep)]
            )
            / keep.sum()
        )
        intervals.append((a, b))
        if a == 0 and b == len(lc.time):
            break
        a, b = max(0, a - 1), min(len(lc.time), b + 1)
    chosen = 0
    while chosen + 1 < len(losses) and losses[chosen + 1] < losses[chosen]:
        chosen += 1
    result = lc.greedy_search(len_deviant=2, sigma=2, direction="negative", max_events=1)
    assert (result.candidates[0]["start"], result.candidates[0]["stop"]) == intervals[chosen]
    mean, std = fit.predict(lc.time, ~result.mask, include_noise=True, errors=lc.flux_err)
    np.testing.assert_allclose(result.mean, mean, atol=1e-9)
    np.testing.assert_allclose(result.std, std, atol=1e-9)


def test_min_remaining_stop_and_boundary_seeds():
    y = np.zeros(12)
    y[:2] = -1
    y[-2:] = 1
    lc = lightcurve(y)
    result = lc.greedy_search(
        growth="threshold", sigma=1.5, grow_sigma=1, len_deviant=2, min_remaining=10
    )
    assert result.mask.sum() == 2
    assert result.settings["stop_reason"] == "min_remaining"
    assert result.candidates[0]["start"] == 0 or result.candidates[0]["stop"] == 12
