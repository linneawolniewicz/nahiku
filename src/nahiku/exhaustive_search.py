"""Exhaustive conditional interval detection."""

import time
import numpy as np
from .gp_helpers import condition, condition_precision
from .search import interval_score, validate_direction, directional
from .results import SearchResult
from .numerics import stable_cholesky


def exhaustive_search(
    owner,
    *,
    min_points=3,
    max_points=20,
    step=2,
    size_step=2,
    metric="pvalue",
    threshold=1e-5,
    top_k=None,
    direction="negative",
    acceleration=True,
    refit=False,
    gap_factor=5.0,
    backend="torch",
    warm_initialization=True,
):
    """Identify anomalous intervals in a time series using Gaussian Processes.

    Method:
        1. List every candidate contiguous interval based on the configured
           minimum/maximum lengths and window steps. For each length, also test
           the window ending at the final observation, even if the start-position
           step would omit it. Skip any window containing a time gap greater
           than ``gap_factor`` times the median spacing between observations.
        2. Use the GP already fitted to the entire time series with ``fit_gp``.
           Optionally cache its full precision matrix for accelerated conditioning.
        3. For each candidate interval, compute its conditional observation
           distribution from a GP conditioned on the rest of the data. With
           ``refit=True``, fit the retained observations independently for each
           interval; warm initialization uses the full-data hyperparameters.
        4. Evaluate the selected test metric on the held-out interval, retaining
           its full covariance for joint metrics. Precision acceleration and
           direct conditioning implement the same conditional distribution at
           fixed hyperparameters; refitting requires ``acceleration=False``.
        5. Filter by the requested residual sign and flag intervals passing the
           metric threshold, or select ``top_k`` ranked non-overlapping intervals.
           Smaller p-value/log-likelihood scores and larger loss/distance scores
           indicate stronger anomalies; the metric definitions specify the direction.
        6. Merge selected intervals into a detection mask. Compute the final
           display prediction outside that union using the full-data fitted
           hyperparameters, even when individual candidate scores used refits.
        7. Return an independent ``SearchResult`` containing every tested interval
           and score, the detection mask, final prediction, and numerical diagnostics.

    Selected windows retain their exact scored mean, observation covariance and
    uncertainty, plus full-grid conditional predictions in ``conditional_predictions``.
    ``result.plot()`` shows representatives; ``selection="all"`` shows all selected
    windows, and ``event_index=0`` shows one representative over the full light curve.
    These plots need no live GP or repeated refit. Storage and full-grid prediction
    costs grow with the number of selected windows. Refitted searches evaluate
    eligible full-grid predictions while each candidate model is available.

    Torch and NumPy support direct and accelerated conditioning. Failed precision
    conditioning retries directly; covariance jitter is bounded and recorded.

    :param owner: Light-curve object with a current GP fit; supplies observations and provenance.
    :type owner: Nahiku
    :param min_points: Minimum candidate interval length in samples.
    :type min_points: int
    :param max_points: Maximum candidate length in samples; smaller than the full light curve.
    :type max_points: int
    :param step: Positive sample spacing between candidate start positions; also include the
        terminal window.
    :type step: int
    :param size_step: Positive spacing between candidate lengths; also include max_points.
    :type size_step: int
    :param metric: "pvalue", "ll" (joint held-out log density), or "mll" (mean pointwise
        marginal held-out log density): smaller is stronger. "mahalanobis", "nlpd",
        "msll" (pointwise loss relative to a retained-data Gaussian), or "rmse": larger
        is stronger. MLL differs from the retained prior MLL used in greedy growth.
    :type metric: str
    :param threshold: Inclusive score cutoff; set to None when using top_k. P-value cutoffs lie in
        [0, 1].
    :type threshold: float or None
    :param top_k: Maximum number of ranked non-overlapping intervals; requires threshold=None.
    :type top_k: int or None
    :param direction: Sign of the mean residual across a candidate window: "negative" (default),
            "positive", or "both". Individual samples need not all have that sign.
    :type direction: str
    :param acceleration: Reuse the full precision matrix for interval conditioning; requires
        refit=False.
    :type acceleration: bool
    :param refit: Retrain GP hyperparameters on retained samples instead of using the original
        fitted parameters.
    :type refit: bool
    :param gap_factor: Positive multiplier of median cadence defining gaps that detections must not
        bridge.
    :type gap_factor: float
    :param backend: "torch" (default) conditions on the fit device; "numpy" uses SciPy on CPU.
        Scores are returned and evaluated as NumPy values with either backend.
    :type backend: str
    :param warm_initialization: Start each interval refit from the full-data fitted parameters,
        independently of other intervals. False uses fresh configured initialization.
        Optimizer state is new each time. Only used when refit=True.
    :type warm_initialization: bool
    :returns: Independent snapshot of detections, predictions, scores, settings, and provenance.
    :rtype: SearchResult
    """
    # Validate interval settings and enforce mutually exclusive selection modes.
    start_clock = time.perf_counter()
    validate_direction(direction)
    if backend not in ("numpy", "torch"):
        raise ValueError("backend must be numpy or torch")
    if not isinstance(warm_initialization, bool):
        raise ValueError("warm_initialization must be bool")
    n = len(owner.time)
    for value in (min_points, max_points, step, size_step):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("Window lengths and steps must be positive integers")
    if min_points > max_points or max_points >= n:
        raise ValueError("Require min_points <= max_points < number of samples")
    if metric not in ("pvalue", "mahalanobis", "nlpd", "rmse", "msll", "mll", "ll"):
        raise ValueError("Unsupported metric")
    if gap_factor <= 0:
        raise ValueError("gap_factor must be positive")
    if (threshold is None) == (top_k is None):
        raise ValueError("Supply exactly one of threshold or top_k (set threshold=None for top_k)")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 1):
        raise ValueError("top_k must be a positive integer")
    if threshold is not None and (
        not np.isfinite(threshold) or (metric == "pvalue" and not 0 <= threshold <= 1)
    ):
        raise ValueError("Invalid threshold")
    if refit and acceleration:
        raise ValueError("Refitting requires acceleration=False")
    # Prepare the fitted covariance, optional precision cache, and observing-gap mask.
    fit = owner._require_fit()
    numerical_diagnostics = [dict(event, source="base_fit") for event in fit.numerical_diagnostics]
    smaller_is_stronger = metric in ("pvalue", "ll", "mll")
    precision = fit.precision if acceleration and backend == "numpy" else None
    if backend == "torch":
        import torch
        from .torch_backend import TorchConditioner, condition_torch

        # Keep the full covariance and optional precision on the selected device throughout search.
        predictor = TorchConditioner(fit, numerical_diagnostics)
        tensor_options = dict(dtype=fit.torch_dtype, device=fit.device)
        if acceleration:
            factor, predictor.cov = stable_cholesky(
                predictor.cov, numerical_diagnostics, "exhaustive full observation covariance"
            )
            precision = torch.cholesky_inverse(factor)
    gap = np.diff(owner.time) > gap_factor * np.median(np.diff(owner.time))
    candidates = []
    conditional_records = {}
    # Enumerate interval widths and starting positions, including both terminal bounds.
    lengths = sorted(set(range(min_points, max_points + 1, size_step)) | {max_points})
    for width in lengths:
        starts = sorted(set(range(0, n - width + 1, step)) | {n - width})
        for a in starts:
            b = a + width
            if gap[a : b - 1].any():
                continue
            # Hold out each gap-free interval and condition its prediction on all other samples.
            test = np.arange(a, b)
            train = np.r_[0:a, b:n]
            current = (
                owner._refit_subset(train, initial_fit=fit if warm_initialization else None)
                if refit
                else fit
            )
            if refit:
                numerical_diagnostics.extend(
                    dict(event, source=f"refit interval [{a}, {b})")
                    for event in current.numerical_diagnostics
                )
            if backend == "torch":
                if refit:
                    covariance, mean = owner._full_prior(current)
                    prediction, cov = condition_torch(
                        torch.as_tensor(covariance, **tensor_options),
                        torch.as_tensor(mean, **tensor_options),
                        predictor.flux,
                        train,
                        test,
                        diagnostics=numerical_diagnostics,
                    )
                else:
                    prediction, cov = condition_torch(
                        predictor.cov,
                        predictor.mean,
                        predictor.flux,
                        train,
                        test,
                        precision,
                        numerical_diagnostics,
                    )
            elif refit:
                # Evaluate the newly fitted prior at every original timestamp.
                covariance, mean = owner._full_prior(current)
                prediction, cov = condition(
                    covariance, mean, owner.flux, train, test, numerical_diagnostics
                )
            elif acceleration:
                prediction, cov = condition_precision(
                    precision,
                    fit.prior_mean,
                    owner.flux,
                    test,
                    cov=fit.cov,
                    diagnostics=numerical_diagnostics,
                )
            else:
                prediction, cov = condition(
                    fit.cov, fit.prior_mean, owner.flux, train, test, numerical_diagnostics
                )
            # Keep each tested score and residual sign for later ranking and inspection.
            score = interval_score(
                owner.flux[test],
                prediction,
                cov,
                metric,
                reference_flux=owner.flux[train],
                diagnostics=numerical_diagnostics,
            )
            candidates.append(
                dict(
                    start=int(a),
                    stop=int(b),
                    score=score,
                    mean_residual=float(np.mean(owner.flux[test] - prediction)),
                )
            )
            # Preserve the exact scored distribution for potentially selected windows.
            candidate = candidates[-1]
            passes = threshold is None or (score <= threshold if smaller_is_stronger else score >= threshold)
            if passes and directional(candidate["mean_residual"], direction):
                record = dict(candidate, mean=prediction.copy(),
                              std=np.sqrt(np.maximum(np.diag(cov), 0)), covariance=cov.copy())
                # A refitted model must be evaluated before it leaves scope; never refit for plotting.
                if refit:
                    full_predictor = TorchConditioner(current, numerical_diagnostics) if backend == "torch" else current
                    record["full_mean"], record["full_std"] = full_predictor.predict(
                        owner.time, include_noise=True, errors=owner.flux_err)
                conditional_records[(a, b)] = record
    # Filter by residual direction and rank deterministically by score and interval bounds.
    eligible = [c for c in candidates if directional(c["mean_residual"], direction)]
    ranked = sorted(
        eligible,
        key=lambda c: ((c["score"] if smaller_is_stronger else -c["score"]), c["start"], c["stop"]),
    )
    if top_k is not None:
        # Top-k means non-overlapping intervals, rather than near-duplicate windows.
        # Apply the inclusive metric threshold when a top-k selection was not requested.
        chosen = []
        for c in ranked:
            if all(c["stop"] <= d["start"] or d["stop"] <= c["start"] for d in chosen):
                chosen.append(c)
            if len(chosen) == top_k:
                break
    else:
        # Apply the inclusive metric threshold when a top-k selection was not requested.
        chosen = [
            c
            for c in ranked
            if (c["score"] <= threshold if smaller_is_stronger else c["score"] >= threshold)
        ]
    # Merge selected intervals into one sample mask for the final display prediction.
    mask = np.zeros(n, bool)
    for c in chosen:
        mask[c["start"] : c["stop"]] = True
    # Store full-grid predictions only for selected fixed-parameter windows.
    selected_predictions = []
    for candidate in chosen:
        a, b = candidate["start"], candidate["stop"]
        record = conditional_records[(a, b)]
        if not refit:
            retained = np.ones(n, bool)
            retained[a:b] = False
            full_predictor = predictor if backend == "torch" else fit
            record["full_mean"], record["full_std"] = full_predictor.predict(
                owner.time, retained, include_noise=True, errors=owner.flux_err)
        # Use the exact scored distribution within the excluded interval.
        record["full_mean"][a:b] = record["mean"]
        record["full_std"][a:b] = record["std"]
        selected_predictions.append(record)
    # A coherent display fit conditioned outside the union of flagged samples.
    display = predictor if backend == "torch" else fit
    previous_records = len(fit.numerical_diagnostics)
    mean, std = display.predict(owner.time, ~mask, include_noise=True, errors=owner.flux_err)
    numerical_diagnostics.extend(
        dict(event, source="display prediction")
        for event in fit.numerical_diagnostics[previous_records:]
    )
    settings = dict(
        numerical_diagnostics=numerical_diagnostics,
        backend=backend,
        warm_initialization=warm_initialization,
        min_points=min_points,
        max_points=max_points,
        step=step,
        size_step=size_step,
        metric=metric,
        threshold=threshold,
        top_k=top_k,
        direction=direction,
        acceleration=acceleration,
        refit=refit,
        gap_factor=gap_factor,
        display_prediction="fixed hyperparameters conditioned outside detected union",
    )
    # Return a self-contained snapshot of observations, candidates, and search provenance.
    return SearchResult(
        "exhaustive",
        owner.time.copy(),
        owner.flux.copy(),
        mean,
        std,
        mask,
        candidates,
        settings,
        time.perf_counter() - start_clock,
        owner._provenance(),
        conditional_predictions=selected_predictions,
    )
