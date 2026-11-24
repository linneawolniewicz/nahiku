"""Bounded covariance stabilization without modifying learned observation noise."""

import warnings
import numpy as np
import torch
from scipy.linalg import cholesky


class CovarianceError(np.linalg.LinAlgError):
    """A covariance could not be factored within the numerical-jitter budget."""


def record_recovery(diagnostics, **event):
    """Record a numerical adjustment, or warn when no record collector is supplied.

    :param diagnostics: Mutable event list, or None to emit a runtime warning.
    :type diagnostics: list or None
    :param event: Serializable context and adjustment details.
    :type event: dict
    :returns: None.
    :rtype: None
    """
    if diagnostics is None:
        warnings.warn(f"Covariance numerical recovery: {event}", RuntimeWarning, stacklevel=2)
    else:
        diagnostics.append(event)


def stable_cholesky(matrix, diagnostics=None, context="covariance", allow_jitter=True):
    """Symmetrize and factor a matrix with bounded, scale-relative diagonal retries.

    :param matrix: Square float32/float64 NumPy array or Torch tensor.
    :type matrix: numpy.ndarray or torch.Tensor
    :param diagnostics: List receiving symmetry and jitter adjustments; None emits warnings.
    :type diagnostics: list or None
    :param context: Operation or interval identifier included in records and errors.
    :type context: str
    :param allow_jitter: Permit diagonal retries up to 1e-6 times matrix scale. False
        performs only the symmetric, unjittered attempt, as used before direct fallback.
    :type allow_jitter: bool
    :returns: Lower Cholesky factor and the symmetric, possibly adjusted matrix.
    :rtype: tuple
    """
    is_torch = isinstance(matrix, torch.Tensor)
    backend = "torch" if is_torch else "numpy"
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise CovarianceError(f"{context}: covariance must be square")
    finite = bool(torch.isfinite(matrix).all()) if is_torch else bool(np.isfinite(matrix).all())
    if not finite:
        raise CovarianceError(f"{context}: non-finite covariance; check inputs and GP parameters")
    # Reconcile roundoff asymmetry before every factorization, without changing the input.
    asymmetry = (
        float((matrix - matrix.T).abs().max())
        if is_torch and matrix.numel()
        else (float(np.max(np.abs(matrix - matrix.T))) if not is_torch and matrix.size else 0.0)
    )
    symmetric = (matrix + matrix.T) * 0.5
    if asymmetry:
        record_recovery(
            diagnostics,
            action="symmetrize",
            context=context,
            backend=backend,
            max_asymmetry=asymmetry,
        )
    if is_torch:
        eps = torch.finfo(matrix.dtype).eps
        diagonal = torch.diagonal(symmetric)
        scale = float(diagonal.abs().max()) if len(diagonal) else 1.0
        identity = torch.eye(len(matrix), dtype=matrix.dtype, device=matrix.device)
    else:
        eps = np.finfo(matrix.dtype).eps
        diagonal = np.diag(symmetric)
        scale = float(np.max(np.abs(diagonal))) if len(diagonal) else 1.0
        identity = np.eye(len(matrix), dtype=matrix.dtype)
    # A zero diagonal uses the largest matrix entry, or unit scale for an all-zero matrix.
    if scale == 0:
        scale = float(symmetric.abs().max()) if is_torch else float(np.max(np.abs(symmetric)))
        scale = scale or 1.0
    retries = sorted({max(eps, 1e-12), 1e-10, 1e-8, 1e-6}) if allow_jitter else []
    retries = [r for r in retries if eps <= r <= 1e-6]
    for relative in [0.0, *retries]:
        adjusted = symmetric if relative == 0 else symmetric + (relative * scale) * identity
        try:
            factor = torch.linalg.cholesky(adjusted) if is_torch else cholesky(adjusted, lower=True)
        except (np.linalg.LinAlgError, torch.linalg.LinAlgError):
            continue
        if relative:
            record_recovery(
                diagnostics,
                action="jitter",
                context=context,
                backend=backend,
                absolute_jitter=relative * scale,
                relative_jitter=relative,
                scale=scale,
            )
        return factor, adjusted
    # Report spectral context only on failure; never solve an indefinite covariance as a Gaussian.
    minimum = (
        float(torch.linalg.eigvalsh(symmetric).min())
        if is_torch
        else float(np.linalg.eigvalsh(symmetric).min())
    )
    raise CovarianceError(
        f"{context}: {backend} Cholesky failed for shape {tuple(matrix.shape)}, dtype {matrix.dtype}; "
        f"minimum eigenvalue={minimum:.6g}, scale={scale:.6g}, "
        f"maximum numerical jitter={(1e-6 * scale if allow_jitter else 0):.6g}. "
        "Check conditioning, duplicate times, and noise calibration; try float64. "
        "No Gaussian score was computed from an indefinite matrix."
    )


def numpy_factor(matrix, diagnostics=None, context="covariance"):
    """Return a stabilized lower factor in SciPy cho_solve format.

    :param matrix: Square floating-point covariance matrix.
    :type matrix: numpy.ndarray
    :param diagnostics: Adjustment record collector, or None to warn.
    :type diagnostics: list or None
    :param context: Label included in records and failures.
    :type context: str
    :returns: Lower factor and True, suitable for scipy.linalg.cho_solve.
    :rtype: tuple
    """
    factor, _ = stable_cholesky(matrix, diagnostics, context)
    return factor, True
