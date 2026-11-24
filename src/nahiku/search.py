"""Shared detection scoring and event evaluation."""

import numpy as np
from scipy.linalg import cho_solve
from .numerics import stable_cholesky
from scipy.stats import chi2
from scipy.optimize import linear_sum_assignment


def interval_score(observed, mean, covariance, metric="pvalue", reference_flux=None, diagnostics=None):
    """Return a conditional statistic; pvalues decrease with anomaly strength.

    :param observed: Observed flux vector for the tested interval.
    :type observed: numpy.ndarray
    :param mean: Conditional predictive mean for the tested interval.
    :type mean: numpy.ndarray
    :param covariance: Full observation covariance matrix, including measurement noise, in squared
        flux units.
    :type covariance: numpy.ndarray
    :param metric: "pvalue", "ll" (joint log density), or "mll" (mean pointwise marginal
        log density): smaller is stronger. "mahalanobis", "nlpd", "msll", and "rmse":
        larger is stronger. MLL uses observation marginals, not a posterior as a GP prior.
    :type metric: str
    :param reference_flux: Retained training flux defining the MSLL Gaussian baseline.
        Required for msll; its variance is floored at machine epsilon.
    :type reference_flux: numpy.ndarray or None
    :param diagnostics: Numerical adjustment records; None emits warnings.
    :type diagnostics: list or None
    :returns: Conditional anomaly statistic under the selected metric.
    :rtype: float
    """
    # Measure the residual using the full conditional covariance.
    residual = observed - mean
    lower, covariance = stable_cholesky(covariance, diagnostics, "interval score covariance")
    factor = (lower, True)
    squared = float(residual @ cho_solve(factor, residual))
    # Convert residual strength to the requested probability, distance, or loss statistic.
    if metric == "pvalue":
        return float(chi2.sf(max(squared, 0), len(observed)))
    if metric == "mahalanobis":
        return float(np.sqrt(max(squared, 0)))
    if metric in ("nlpd", "ll"):
        logdet = 2 * np.log(np.diag(factor[0])).sum()
        log_density = -0.5 * (squared + logdet + len(observed) * np.log(2 * np.pi))
        return float(log_density if metric == "ll" else -log_density / len(observed))
    if metric in ("mll", "msll"):
        # Marginal scores use each observation variance, while joint scores retain correlations.
        variance = np.diag(covariance)
        point_loss = 0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)
        if metric == "mll":
            return float(-np.mean(point_loss))
        if reference_flux is None or not len(reference_flux):
            raise ValueError("msll requires nonempty retained reference_flux")
        reference_flux = np.asarray(reference_flux)
        baseline_variance = max(float(np.var(reference_flux)), np.finfo(float).eps)
        baseline_loss = 0.5 * (
            np.log(2 * np.pi * baseline_variance)
            + (observed - np.mean(reference_flux)) ** 2 / baseline_variance
        )
        return float(np.mean(point_loss - baseline_loss))
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual**2)))
    raise ValueError("metric must be pvalue, mahalanobis, nlpd, msll, mll, ll, or rmse")


def evaluate_events(detected, truth, buffer=0):
    """Maximum one-to-one overlap matching of half-open events; no double counting.

    :param detected: Detected half-open sample intervals (start, stop).
    :type detected: iterable of tuple
    :param truth: True half-open sample intervals (start, stop).
    :type truth: iterable of tuple
    :param buffer: Nonnegative overlap tolerance in samples, applied to the truth intervals.
    :type buffer: int
    :returns: Event counts, false positives, misses, precision, and recall; empty denominators
        return zero.
    :rtype: dict
    """
    if not isinstance(buffer, (int, np.integer)) or buffer < 0:
        raise ValueError("buffer must be a nonnegative sample count")
    detected, truth = list(detected), list(truth)
    matches = 0
    if detected and truth:
        # Build the allowed event-overlap pairs, including the requested sample tolerance.
        overlap = np.array(
            [[a < d + buffer and c - buffer < b for c, d in truth] for a, b in detected], dtype=int
        )
        # Maximize one-to-one matches so repeated detections cannot inflate recovery.
        rows, cols = linear_sum_assignment(-overlap)
        matches = int(overlap[rows, cols].sum())
    # Report event counts and define empty-set precision and recall as zero.
    return dict(
        matched_events=matches,
        detected_events=len(detected),
        true_events=len(truth),
        false_positives=len(detected) - matches,
        missed_events=len(truth) - matches,
        precision=matches / len(detected) if detected else 0.0,
        recall=matches / len(truth) if truth else 0.0,
    )


def validate_direction(direction):
    """Reject unsupported residual-sign selections.

    :param direction: Allowed residual sign: "negative", "positive", or "both".
    :type direction: str
    :returns: None.
    :rtype: None
    """
    if direction not in ("negative", "positive", "both"):
        raise ValueError("direction must be negative, positive, or both")


def directional(residual, direction):
    """Test whether a residual has the requested sign.

    :param residual: Mean conditional residual whose sign is tested.
    :type residual: float
    :param direction: Allowed residual sign: "negative", "positive", or "both".
    :type direction: str
    :returns: Whether the residual has the selected sign.
    :rtype: bool
    """
    return direction == "both" or (residual < 0 if direction == "negative" else residual > 0)
