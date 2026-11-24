"""Configurable residual seeding and interval growth with conditional GP updates."""

import time
import numpy as np
from scipy.linalg import cho_solve
from .numerics import stable_cholesky, numpy_factor
from .results import SearchResult
from .search import validate_direction


def _growth_loss(owner, fit, retained, mean, std, metric, backend, diagnostics=None):
    """Evaluate a per-sample loss on the retained observations; smaller is better.

    :param owner: Light curve supplying observations and measurement errors.
    :type owner: Nahiku
    :param fit: GP supplying fixed hyperparameters for this trial.
    :type fit: GPFit
    :param retained: Boolean selection on the original observation grid.
    :type retained: numpy.ndarray
    :param mean: Full-grid posterior mean conditioned on retained samples.
    :type mean: numpy.ndarray
    :param std: Full-grid posterior observation standard deviation.
    :type std: numpy.ndarray
    :param metric: "mll", "nlpd", "msll", or "rmse".
    :type metric: str
    :param backend: "torch" or "numpy" for dense Gaussian calculations.
    :type backend: str
    :param diagnostics: Numerical recovery event collector; None emits warnings.
    :type diagnostics: list or None
    :returns: Scalar loss normalized by the retained sample count.
    :rtype: float
    """
    y = owner.flux[retained]
    residual = y - mean[retained]
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual**2)))
    if metric == "msll":
        # Compare pointwise predictive log loss with a constant Gaussian baseline.
        variance = np.maximum(std[retained] ** 2, np.finfo(float).tiny)
        baseline = max(float(np.var(y)), np.finfo(float).tiny)
        return float(
            0.5
            * np.mean(
                np.log(variance / baseline)
                + residual**2 / variance
                - (y - y.mean()) ** 2 / baseline
            )
        )
    # MLL uses the prior observation covariance; NLPD uses the retained posterior.
    if np.array_equal(fit.time, owner.time[retained]):
        covariance, prior = fit.cov, fit.prior_mean
    else:
        covariance = fit.cov[np.ix_(retained, retained)]
        prior = fit.prior_mean[retained]
    residual = y - prior if metric == "mll" else residual
    if backend == "torch":
        import torch

        options = dict(dtype=fit.torch_dtype, device=fit.device)
        matrix = torch.as_tensor(covariance, **options)
        vector = torch.as_tensor(residual, **options)
        factor, _ = stable_cholesky(matrix, diagnostics, "greedy growth covariance")
        if metric == "nlpd":
            noise = np.full(len(y), fit.noise_variance)
            if owner.flux_err is not None:
                noise += owner.flux_err[retained] ** 2
            kernel = matrix - torch.diag(torch.as_tensor(noise, **options))
            matrix = matrix - kernel @ torch.cholesky_solve(kernel, factor)
            factor, _ = stable_cholesky(
                matrix, diagnostics, "greedy growth posterior covariance"
            )
        quadratic = vector @ torch.cholesky_solve(vector[:, None], factor)[:, 0]
        logdet = 2 * torch.log(torch.diagonal(factor)).sum()
        return float((quadratic + logdet + len(y) * np.log(2 * np.pi)).item() / (2 * len(y)))
    factor = numpy_factor(covariance, diagnostics, "greedy growth covariance")
    if metric == "nlpd":
        noise = np.full(len(y), fit.noise_variance)
        if owner.flux_err is not None:
            noise += owner.flux_err[retained] ** 2
        kernel = covariance - np.diag(noise)
        covariance = covariance - kernel @ cho_solve(factor, kernel)
        factor = numpy_factor(covariance, diagnostics, "greedy growth posterior covariance")
    return float(
        (
            residual @ cho_solve(factor, residual)
            + 2 * np.log(np.diag(factor[0])).sum()
            + len(y) * np.log(2 * np.pi)
        )
        / (2 * len(y))
    )


def greedy_search(
    owner,
    *,
    sigma=3.0,
    grow_sigma=2.0,
    max_events=None,
    direction="both",
    refit=False,
    gap_factor=None,
    backend="torch",
    seed_statistic="residual",
    len_deviant=1,
    growth="metric",
    growth_metric="mll",
    expansion_param=1,
    warm_initialization=True,
    update_threshold=False,
    min_remaining=3,
):
    """Identify anomalous intervals in a time series using Gaussian Processes.

    Method:
        1. Use the GP regression already fitted to the time series with ``fit_gp``.
        2. Find the strongest qualifying outlier interval of length ``len_deviant``.
           Rank qualifying intervals by their minimum residual strength, using
           ordinary posterior residuals or standardized leave-one-out residuals.
        3. Exclude the outlier interval and redo the regression prediction. With
           ``refit=True``, retrain the GP on retained observations; warm initialization
           starts from the preceding accepted hyperparameters by default.
        4. For ``growth="metric"``, expand the outlier interval in both directions
           by ``expansion_param`` and repeat step 3. Evaluate the chosen fitting
           metric on the retained observations.
        5. Repeat step 4 as long as the metric improves. Keep the last improving
           interval. Alternatively, ``growth="threshold"`` extends through same-sign
           neighbors above ``grow_sigma``, without refitting at each extension.
        6. Flag the accepted interval and exclude its observations from later seeds.
           Apply the selected sign restriction and optional observing-gap barrier.
        7. Repeat steps 2–6 while a qualifying interval remains above ``sigma``,
           unless ``max_events`` or ``min_remaining`` stops the search. Residual
           thresholds stay fixed unless ``update_threshold=True``; LOO uses its
           current conditional uncertainty.
        8. Return the final regression, residuals, and accepted intervals in an
           independent ``SearchResult``. Greedy candidates exclude rejected growth
           trials; adjacent accepted intervals can merge into one reported event.

    Both conditioning backends use the same detection rules. Numerical recovery
    is bounded and recorded separately from learned observation noise.

    :param owner: Light curve with a current fitted GP.
    :type owner: Nahiku
    :param sigma: Seed threshold multiplier for residual scatter or standardized LOO residuals.
    :type sigma: float
    :param grow_sigma: Neighbor threshold used only for threshold growth; positive and no greater than sigma. Heterogeneous measurement errors contribute to its scale.
    :type grow_sigma: float
    :param max_events: Maximum seeded events; None runs until no qualifying seed remains,
        subject to min_remaining. Adjacent seeded events can merge in the output mask.
    :type max_events: int or None
    :param direction: "both" compares absolute deviations; "negative" or "positive" selects a sign.
    :type direction: str
    :param refit: Optimize hyperparameters for each candidate exclusion in metric growth,
        or after each accepted threshold-grown event.
    :type refit: bool
    :param gap_factor: Stop seeds and growth across gaps exceeding this multiple of median
        cadence; None disables the gap barrier.
    :type gap_factor: float or None
    :param backend: "torch" (default, on the fit device) or "numpy" (SciPy on CPU).
    :type backend: str
    :param seed_statistic: "residual" uses ordinary posterior residuals and
        sqrt(error**2 + var(abs(residual))); "loo" uses GP leave-one-out z scores.
    :type seed_statistic: str
    :param len_deviant: Number of consecutive qualifying samples required for a seed;
        rank windows by their minimum directional residual strength.
    :type len_deviant: int
    :param growth: "metric" expands exclusions while retained-data loss improves;
        "threshold" extends through same-sign neighbors above grow_sigma.
    :type growth: str
    :param growth_metric: Retained-data loss for metric growth: "mll" (negative normalized
        marginal log likelihood), "nlpd" (joint predictive loss), "msll", or "rmse".
    :type growth_metric: str
    :param expansion_param: Samples added at each available edge per metric-growth trial.
    :type expansion_param: int
    :param warm_initialization: Start refits from the preceding accepted GP parameters.
        False uses the configured initial parameters. Optimizer state is always new.
    :type warm_initialization: bool
    :param update_threshold: Recompute residual scatter after each accepted event; False
        freezes initial scatter. Only available for seed_statistic="residual". LOO always
        uses its current conditional uncertainty and the fixed sigma multiplier.
    :type update_threshold: bool
    :param min_remaining: Minimum observations retained for conditioning and refitting; an integer of at least two, default three.
    :type min_remaining: int
    :returns: Independent predictions, detections, settings, stop reason, and provenance.
    :rtype: SearchResult
    """
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be 'numpy' or 'torch'")
    validate_direction(direction)
    if seed_statistic not in ("residual", "loo") or growth not in ("metric", "threshold"):
        raise ValueError(
            "seed_statistic must be residual or loo; growth must be metric or threshold"
        )
    if growth_metric not in ("mll", "nlpd", "msll", "rmse"):
        raise ValueError("growth_metric must be mll, nlpd, msll, or rmse")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    if growth == "threshold" and (not np.isfinite(grow_sigma) or not 0 < grow_sigma <= sigma):
        raise ValueError("Require 0 < grow_sigma <= sigma")
    for name, value, minimum in (
        ("len_deviant", len_deviant, 1),
        ("expansion_param", expansion_param, 1),
        ("min_remaining", min_remaining, 2),
        ("max_events", max_events, 1),
    ):
        if name == "max_events" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if gap_factor is not None and (not np.isfinite(gap_factor) or gap_factor <= 0):
        raise ValueError("gap_factor must be positive and finite, or None")
    for flag in (refit, warm_initialization, update_threshold):
        if not isinstance(flag, bool):
            raise ValueError("refit, warm_initialization, and update_threshold must be bool")
    if update_threshold and seed_statistic != "residual":
        raise ValueError("update_threshold applies only to the residual seed statistic")
    # Keep original indices throughout and start with an independent event mask.
    clock = time.perf_counter()
    base = owner._require_fit()
    numerical_diagnostics = [dict(event, source="base_fit") for event in base.numerical_diagnostics]
    from .torch_backend import TorchConditioner

    n = len(owner.time)
    mask = np.zeros(n, bool)
    gap = (
        np.zeros(n - 1, bool)
        if gap_factor is None
        else (np.diff(owner.time) > gap_factor * np.median(np.diff(owner.time)))
    )
    candidates = []
    settings = dict(
        sigma=sigma,
        grow_sigma=grow_sigma,
        max_events=max_events,
        direction=direction,
        refit=refit,
        gap_factor=gap_factor,
        backend=backend,
        seed_statistic=seed_statistic,
        len_deviant=len_deviant,
        growth=growth,
        growth_metric=growth_metric,
        expansion_param=expansion_param,
        warm_initialization=warm_initialization,
        update_threshold=update_threshold,
        min_remaining=min_remaining,
    )

    def evaluate(excluded, previous):
        """Condition or refit one exclusion trial without mutating the accepted fit.

        :param excluded: Boolean mask on the full observation grid.
        :type excluded: numpy.ndarray
        :param previous: Accepted GP supplying warm-start parameters.
        :type previous: GPFit
        :returns: Fitted GP, predictor, training mask, full-grid mean and uncertainty.
        :rtype: tuple
        """
        fit = (
            owner._refit_subset(
                np.flatnonzero(~excluded), initial_fit=previous if warm_initialization else None
            )
            if refit and excluded.any()
            else base
        )
        if fit is not base:
            numerical_diagnostics.extend(
                dict(event, source="refit") for event in fit.numerical_diagnostics
            )
        previous_records = len(fit.numerical_diagnostics)
        predictor = fit if backend == "numpy" else TorchConditioner(fit, numerical_diagnostics)
        training = None if refit and excluded.any() else ~excluded
        mean, std = predictor.predict(
            owner.time, training, include_noise=True, errors=owner.flux_err
        )
        numerical_diagnostics.extend(
            dict(event, source="prediction")
            for event in fit.numerical_diagnostics[previous_records:]
        )
        return fit, predictor, training, mean, std

    state = evaluate(mask, base)
    initial_scatter = float(np.var(np.abs(owner.flux - state[3])))
    threshold_history = []
    stop_reason = "threshold"
    while True:
        if max_events is not None and len(candidates) >= max_events:
            stop_reason = "max_events"
            break
        if (~mask).sum() - len_deviant < min_remaining:
            stop_reason = "min_remaining"
            break
        fit, predictor, training, mean, std = state
        available = np.flatnonzero(~mask)
        # Match the significance scale to the selected seed statistic.
        signed = owner.flux - mean
        if seed_statistic == "loo":
            if backend == "torch":
                signed[available] = predictor.loo(training)
            else:
                covariance, prior = owner._full_prior(fit)
                block = covariance[np.ix_(available, available)]
                precision = cho_solve(
                    numpy_factor(block, numerical_diagnostics, "greedy LOO retained covariance"),
                    np.eye(len(available), dtype=block.dtype),
                )
                signed[available] = (
                    precision @ (owner.flux[available] - prior[available])
                ) / np.sqrt(np.diag(precision))
            scale = np.ones(n)
        else:
            scatter = (
                float(np.var(np.abs(signed[available]))) if update_threshold else initial_scatter
            )
            errors = np.zeros(n) if owner.flux_err is None else owner.flux_err
            scale = np.sqrt(np.maximum(errors**2 + scatter, np.finfo(fit.numpy_dtype).tiny))
            threshold_history.append(scatter)
        directional = (
            -signed
            if direction == "negative"
            else signed
            if direction == "positive"
            else np.abs(signed)
        )
        # Require an entire contiguous window to qualify, avoiding centered-window ambiguity.
        windows = np.lib.stride_tricks.sliding_window_view(directional, len_deviant)
        eligible = np.lib.stride_tricks.sliding_window_view(
            (~mask) & (directional >= sigma * scale), len_deviant
        ).all(axis=1)
        if len_deviant > 1:
            eligible &= ~np.lib.stride_tricks.sliding_window_view(gap, len_deviant - 1).any(axis=1)
        strengths = np.where(eligible, windows.min(axis=1), -np.inf)
        a = int(np.argmax(strengths))
        if not np.isfinite(strengths[a]):
            break
        b = a + len_deviant
        score = float(np.min(directional[a:b] / scale[a:b]))
        # Threshold growth uses this iteration's fixed scale and the strongest seed's sign.
        if growth == "threshold":
            sign = np.sign(signed[a + np.argmax(directional[a:b])])
            while (
                a > 0
                and not gap[a - 1]
                and not mask[a - 1]
                and sign * signed[a - 1] >= grow_sigma * scale[a - 1]
                and (~mask).sum() - (b - a) - 1 >= min_remaining
            ):
                a -= 1
            while (
                b < n
                and not gap[b - 1]
                and not mask[b]
                and sign * signed[b] >= grow_sigma * scale[b]
                and (~mask).sum() - (b - a) - 1 >= min_remaining
            ):
                b += 1
            trial = mask.copy()
            trial[a:b] = True
            state = evaluate(trial, fit)
        else:
            # Evaluate the seed exclusion, then retain only strictly improving expansions.
            trial = mask.copy()
            trial[a:b] = True
            state = evaluate(trial, fit)
            loss = _growth_loss(
                owner,
                state[0],
                ~trial,
                state[3],
                state[4],
                growth_metric,
                backend,
                numerical_diagnostics,
            )
            while True:
                left, right = a, b
                for _ in range(expansion_param):
                    if left > 0 and not gap[left - 1] and not mask[left - 1]:
                        left -= 1
                    if right < n and not gap[right - 1] and not mask[right]:
                        right += 1
                proposed = mask.copy()
                proposed[left:right] = True
                if (left, right) == (a, b) or (~proposed).sum() < min_remaining:
                    break
                candidate = evaluate(proposed, state[0])
                new_loss = _growth_loss(
                    owner,
                    candidate[0],
                    ~proposed,
                    candidate[3],
                    candidate[4],
                    growth_metric,
                    backend,
                    numerical_diagnostics,
                )
                if not np.isfinite(new_loss) or new_loss >= loss:
                    break
                a, b, state, loss = left, right, candidate, new_loss
        # Commit only the accepted interval and its corresponding conditional model.
        candidates.append(
            dict(
                start=a,
                stop=b,
                score=score,
                mean_residual=float(np.mean(owner.flux[a:b] - mean[a:b])),
            )
        )
        mask[a:b] = True
    settings.update(
        stop_reason=stop_reason,
        residual_variance_history=threshold_history,
        numerical_diagnostics=numerical_diagnostics,
    )
    return SearchResult(
        "greedy",
        owner.time.copy(),
        owner.flux.copy(),
        state[3],
        state[4],
        mask,
        candidates,
        settings,
        time.perf_counter() - clock,
        owner._provenance(),
    )
