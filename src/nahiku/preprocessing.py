"""Timestamp-aware period estimation and iterative sinusoid removal.

The default delegates to Dan Hey's Balmung implementation. Nāhiku's separate
weighted, bounded-frequency implementation is available with method='weighted'.
"""

from dataclasses import dataclass
import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import minimize_scalar


@dataclass
class PrewhiteningResult:
    """Removed sinusoidal components and before/after periodograms.

    Amplitudes, model, and residuals use input flux units. PSD-normalized power
    uses squared flux units without error weights and is dimensionless when flux
    is weighted by supplied measurement errors.
    
    Phases refer to ``time_origin`` and the convention
    ``amplitude*cos(2*pi*frequency*(time-time_origin)+phase)``.
    
    :param frequencies: Removed component frequencies in cycles per day.
    :type frequencies: numpy.ndarray
    :param amplitudes: Removed component amplitudes in input flux units.
    :type amplitudes: numpy.ndarray
    :param phases: Cosine phases in radians relative to time_origin.
    :type phases: numpy.ndarray
    :param model: Sum of removed sinusoids on the complete observation grid.
    :type model: numpy.ndarray
    :param residuals: Input flux minus the removed model; the continuum is preserved.
    :type residuals: numpy.ndarray
    :param frequency_grid: Periodogram grid in cycles per day.
    :type frequency_grid: numpy.ndarray
    :param power_before: Initial Lomb–Scargle power in PSD normalization.
    :type power_before: numpy.ndarray
    :param power_after: Residual Lomb–Scargle power on frequency_grid.
    :type power_after: numpy.ndarray
    :param stopping_reason: Termination condition: component_limit, minimum_snr, or
        optimizer_failure (weighted method only; Balmung optimizer errors propagate).
    :type stopping_reason: str
    :param time_origin: First input timestamp in days, used as the phase origin.
    :type time_origin: float
    :param method: Prewhitening algorithm: "balmung" or "weighted".
    :type method: str
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    model: np.ndarray
    residuals: np.ndarray
    frequency_grid: np.ndarray
    power_before: np.ndarray
    power_after: np.ndarray
    stopping_reason: str
    time_origin: float
    method: str = "balmung"

    def plot(self, axes=None):
        """Return a figure and two axes comparing input/residual flux and spectra.

        The flux panel uses sample indices because this result stores a phase
        origin rather than the complete timestamp array.

        :param axes: Two axes for the diagnostic panels; None creates a new figure.
        :type axes: sequence of matplotlib.axes.Axes or None
        :returns: Figure and axes handles; no display is triggered.
        :rtype: tuple
        """
        import matplotlib.pyplot as plt

        if axes is None:
            _, axes = plt.subplots(2, 1, figsize=(9, 5))
        # Compare the original and residual series alongside their frequency spectra.
        axes[0].plot(self.residuals + self.model, lw=0.6, label="before")
        axes[0].plot(self.residuals, lw=0.6, label="after")
        axes[0].set(xlabel="Sample", ylabel="Flux")
        axes[1].plot(self.frequency_grid, self.power_before, label="before")
        axes[1].plot(self.frequency_grid, self.power_after, label="after")
        axes[1].set(xlabel="Frequency [1/day]", ylabel="Lomb–Scargle power")
        for ax in axes:
            ax.legend()
        axes[0].figure.tight_layout()
        return axes[0].figure, axes


def prewhiten(
    time,
    flux,
    errors,
    *,
    fmin,
    fmax,
    max_components=20,
    minimum_snr=4.0,
    samples_per_peak=None,
    fit_mask=None,
    method="balmung",
):
    """Remove successively fitted sinusoids while retaining the continuum level.

    :param time: Timestamps in days, on the input time reference.
    :type time: numpy.ndarray
    :param flux: Flux samples in consistent linear units, aligned with time.
    :type flux: numpy.ndarray
    :param errors: Positive one-sigma measurement errors in flux units; None omits known errors.
    :type errors: numpy.ndarray or None
    :param fmin: Positive lower frequency bound in cycles per day.
    :type fmin: float
    :param fmax: Upper frequency bound in cycles per day; greater than fmin.
    :type fmax: float
    :param max_components: Maximum number of sinusoidal components to remove.
    :type max_components: int
    :param minimum_snr: Amplitude threshold multiplier; fixed from the initial spectrum
        for Balmung, recomputed from residual power for the weighted method.
    :type minimum_snr: float
    :param samples_per_peak: None selects Balmung's fixed fivefold frequency sampling or
        ten samples per peak for the weighted method. Balmung accepts only None or 5;
        the weighted method accepts integers of at least two. Also sets diagnostic sampling.
    :type samples_per_peak: int or None
    :param fit_mask: Boolean mask selecting samples used to fit sinusoids; None uses all samples.
        The fitted signal is subtracted at all times.
    :type fit_mask: array-like or None
    :param method: "balmung" (default) delegates to Dan Hey’s unweighted joint sinusoid fitting and a
        fixed initial amplitude threshold. "weighted" uses uncertainty-weighted,
        locally bounded frequency fitting and a residual-spectrum threshold.
        Balmung retains supplied errors but does not use them in fitting, and
        rejects fit_mask; use "weighted" to protect samples.
    :type method: str
    :returns: Component parameters and full-grid model/residual diagnostics.
    :rtype: PrewhiteningResult
    """
    if method not in ("balmung", "weighted"):
        raise ValueError("method must be 'balmung' or 'weighted'")
    if samples_per_peak is None:
        samples_per_peak = 5 if method == "balmung" else 10
    if isinstance(samples_per_peak, bool) or not isinstance(samples_per_peak, (int, np.integer)):
        raise ValueError("samples_per_peak must be an integer or None")
    if method == "balmung" and samples_per_peak != 5:
        raise ValueError("Balmung uses fixed fivefold sampling; use samples_per_peak=None or 5, or method='weighted'")
    if method == "balmung" and fit_mask is not None:
        raise ValueError("fit_mask requires method='weighted'; Balmung uses all samples")
    if not 0 < fmin < fmax or max_components < 1 or minimum_snr <= 0 or samples_per_peak < 2:
        raise ValueError(
            "Require 0 < fmin < fmax, positive stopping settings and samples_per_peak >= 2"
        )
    # Restrict component fitting to selected samples while keeping the full output grid.
    mask = np.ones(len(time), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    if mask.shape != time.shape or mask.sum() < 4:
        raise ValueError("fit_mask must match time and retain at least four samples")
    # Shift the time origin for numerical stability and initialize residual bookkeeping.
    origin = float(time[0])
    x = time - origin
    residual = flux.copy()
    total = np.zeros_like(flux)
    err = None if errors is None or method == "balmung" else errors[mask]

    def spectrum(y):
        """Evaluate the masked, uncertainty-weighted residual periodogram.

        :param y: Full-grid residual flux; the enclosing fit_mask selects periodogram samples.
        :type y: numpy.ndarray
        :returns: Frequency grid and power spectral density.
        :rtype: tuple
        """
        # Weight the masked periodogram by measurement uncertainties when available.
        ls = LombScargle(x[mask], y[mask], dy=err)
        # A constant series has no oscillatory power and needs no numerical periodogram.
        if np.ptp(y[mask]) == 0:
            grid = ls.autofrequency(
                minimum_frequency=fmin, maximum_frequency=fmax,
                samples_per_peak=samples_per_peak,
            )
            return grid, np.zeros_like(grid)
        return ls.autopower(
            minimum_frequency=fmin,
            maximum_frequency=fmax,
            samples_per_peak=samples_per_peak,
            normalization="psd",
        )

    # Save the initial spectrum before iteratively extracting its strongest components.
    grid, before = spectrum(residual)
    if method == "balmung":
        from .balmung import Balmung

        # Use Balmung's original joint fitting and fixed initial amplitude threshold.
        engine = Balmung(x, flux)
        if np.ptp(flux) == 0:
            reason = "minimum_snr"
        else:
            engine.prewhiten(
                fmin=fmin, fmax=fmax, minimum_snr=minimum_snr,
                maxiter=max_components, diagnose=False,
            )
            # Balmung stops at its fixed amplitude threshold or the component limit.
            reason = "component_limit" if len(engine.removed) >= max_components else "minimum_snr"
        components = np.asarray(engine.removed, dtype=float).reshape(-1, 3)
        # Reconstruct only removed oscillations, preserving the input continuum level.
        for frequency, amplitude, phase in components:
            total += engine.model(x, frequency, amplitude, phase)
        residual = flux - total
        _, after = spectrum(residual)
        return PrewhiteningResult(
            components[:, 0], components[:, 1], components[:, 2], total, residual,
            grid, before, after, reason, origin, method,
        )
    frequencies, amplitudes, phases = [], [], []
    reason = "component_limit"
    for _ in range(max_components):
        grid, power = spectrum(residual)
        # Stop when the strongest remaining peak fails the amplitude-ratio threshold.
        index = int(np.argmax(power))
        if np.sqrt(max(power[index], 0) / max(np.median(power), np.finfo(float).eps)) < minimum_snr:
            reason = "minimum_snr"
            break

        def solve(frequency):
            """Fit an offset and sinusoid at a trial frequency by weighted least squares.

            :param frequency: Trial sinusoid frequency in cycles per day.
            :type frequency: float
            :returns: Weighted residual sum of squares and offset/cosine/sine coefficients.
            :rtype: tuple
            """
            # Solve an offset plus sine and cosine terms at the trial frequency.
            angle = 2 * np.pi * frequency * x[mask]
            design = np.c_[np.ones(mask.sum()), np.cos(angle), np.sin(angle)]
            weights = np.ones(mask.sum()) if err is None else 1 / err
            coefficients = np.linalg.lstsq(
                design * weights[:, None], residual[mask] * weights, rcond=None
            )[0]
            loss = np.sum(((residual[mask] - design @ coefficients) * weights) ** 2)
            return loss, coefficients

        # Refine the peak frequency between its adjacent grid points.
        lower, upper = grid[max(index - 1, 0)], grid[min(index + 1, len(grid) - 1)]
        best = minimize_scalar(
            lambda f: solve(f)[0], bounds=(lower, upper), method="bounded", options={"xatol": 1e-10}
        )
        if not best.success:
            reason = "optimizer_failure"
            break
        frequency = float(best.x)
        _, coefficients = solve(frequency)
        _, cosine, sine = coefficients
        component = cosine * np.cos(2 * np.pi * frequency * x) + sine * np.sin(
            2 * np.pi * frequency * x
        )
        # Subtract only the oscillation, preserving the fitted continuum offset.
        residual -= component
        total += component
        frequencies.append(frequency)
        amplitudes.append(float(np.hypot(cosine, sine)))
        phases.append(float(np.arctan2(-sine, cosine)))
    # Return component parameters and spectra together with full-grid residuals.
    _, after = spectrum(residual)
    return PrewhiteningResult(
        np.array(frequencies),
        np.array(amplitudes),
        np.array(phases),
        total,
        residual,
        grid,
        before,
        after,
        reason,
        origin,
        method,
    )
