"""Central light-curve workflow with explicit preprocessing and independent results."""

from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from astropy.timeseries import LombScargle
from scipy.ndimage import gaussian_filter1d
from .gp_helpers import GPFit
from .results import events_from_mask
from .preprocessing import prewhiten as _prewhiten
from .search import evaluate_events


def _readonly(values):
    """Copy values into an immutable floating-point array.

    :param values: Values to copy into a read-only floating-point array.
    :type values: array-like
    :returns: Read-only copy of the input values.
    :rtype: numpy.ndarray
    """
    result = np.array(values, dtype=float, copy=True)
    result.flags.writeable = False
    return result


class Nahiku:
    """A light curve with original observations, working data, preprocessing history,
    a GP fit, and independent search results.

    :param time: Finite, strictly increasing timestamps in days, with at least three samples.
    :type time: array-like
    :param flux: Finite flux matching time, in consistent linear units.
    :type flux: array-like
    :param flux_err: Positive one-sigma measurement errors matching flux, or None.
    :type flux_err: array-like or None
    :param metadata: JSON-serializable observation provenance, copied into the object.
    :type metadata: dict or None

    Notes
    -----
    Construction does not normalize, estimate a period, download data, or fit.
    Arrays are read-only. Data-changing methods update working data, preserve ``original_flux``
    and ``original_flux_err``, and invalidate fitted/search state.
    """

    def __init__(self, time, flux, flux_err=None, metadata=None):
        """Validate observations and initialize independent working and original arrays.

        :param time: Finite, strictly increasing one-dimensional timestamps in days; at least three
            samples.
        :type time: array-like
        :param flux: Finite one-dimensional flux matching time; construction does not preprocess it.
        :type flux: array-like
        :param flux_err: Positive one-sigma measurement errors in flux units; None omits known
            errors.
        :type flux_err: array-like or None
        :param metadata: JSON-serializable observation provenance; copied into the object.
        :type metadata: dict or None
        :returns: None.
        :rtype: None
        """
        # Validate the observation grid before creating any analysis state.
        time, flux = np.asarray(time, float), np.asarray(flux, float)
        if time.ndim != 1 or flux.shape != time.shape or len(time) < 3:
            raise ValueError("time and flux must be matching 1D arrays with at least three samples")
        if not np.isfinite(time).all() or not np.isfinite(flux).all() or np.any(np.diff(time) <= 0):
            raise ValueError("Require finite flux and finite, strictly increasing time")
        if flux_err is not None:
            flux_err = np.asarray(flux_err, float)
            if (
                flux_err.shape != time.shape
                or not np.isfinite(flux_err).all()
                or np.any(flux_err <= 0)
            ):
                raise ValueError("flux_err must match time and contain finite positive errors")
        # Keep immutable originals separate from the working observations.
        self._time, self._original_flux = _readonly(time), _readonly(flux)
        self._original_flux_err = None if flux_err is None else _readonly(flux_err)
        self._flux = _readonly(flux)
        self._flux_err = None if flux_err is None else _readonly(flux_err)
        self.metadata = deepcopy(metadata or {})
        json.dumps(self.metadata, allow_nan=False)
        # Start with no preprocessing, fitted model, or detected events.
        self.history = []
        self.period = None
        self.period_result = None
        self.prewhitening = None
        self.results = {}
        self._fit = None
        self._fit_options = None
        self.truth_mask = np.zeros(len(time), bool)

    @property
    def time(self):
        """Read-only timestamps in days, unchanged by preprocessing.

        :returns: Read-only observation array.
        :rtype: numpy.ndarray
        """
        return self._time

    @property
    def flux(self):
        """Read-only working flux.

        :returns: Read-only observation array.
        :rtype: numpy.ndarray
        """
        return self._flux

    @property
    def flux_err(self):
        """Working one-sigma errors, or None.

        :returns: Read-only one-sigma errors, or None if unavailable.
        :rtype: numpy.ndarray or None
        """
        return self._flux_err

    @property
    def original_flux(self):
        """Original observations, before standardization, prewhitening, or injection.

        :returns: Read-only observation array.
        :rtype: numpy.ndarray
        """
        return self._original_flux

    @property
    def original_flux_err(self):
        """Original measurement uncertainties, or None.

        :returns: Read-only one-sigma errors, or None if unavailable.
        :rtype: numpy.ndarray or None
        """
        return self._original_flux_err

    @property
    def fit_diagnostics(self):
        """Loss history and stopping settings of the current fit (copy).

        :returns: Copied loss history and optimization settings.
        :rtype: dict
        """
        fit = self._require_fit()
        return dict(losses=list(fit.losses), **deepcopy(fit.settings))

    def _invalidate(self):
        """Clear GP, search, and period state after working observations change.

        :returns: None.
        :rtype: None
        """
        self._fit = None
        self._fit_options = None
        self.results = {}
        self.period = None
        self.period_result = None

    def _provenance(self):
        """Collect copied processing history, fitted settings, and dependency versions.

        :returns: Independent serializable analysis provenance.
        :rtype: dict
        """
        # Copy provenance so later changes cannot alter saved results.
        metadata = deepcopy(self.metadata)
        metadata["preprocessing"] = deepcopy(self.history)
        if self._fit is not None:
            import platform
            from importlib.metadata import version

            # Record learned parameters and the environment needed to reproduce the fit.
            metadata["gp_fit"] = deepcopy(self._fit.settings)
            metadata["gp_fit"]["learned_parameters"] = {
                name: value.detach().cpu().numpy().tolist()
                for name, value in self._fit.model.named_parameters()
            }
            metadata["environment"] = dict(
                python=platform.python_version(),
                platform=platform.platform(),
                dependencies={
                    name: version(name)
                    for name in ("nahiku", "numpy", "scipy", "astropy", "torch", "gpytorch")
                },
            )
        return metadata

    def copy(self):
        """Return a deep independent copy, including current preprocessing and results.

        :returns: Independent new light-curve object.
        :rtype: Nahiku
        """
        return deepcopy(self)

    def reset(self):
        """Restore original observations and clear derived state and injection labels.

        :returns: This object, for chaining.
        :rtype: Nahiku
        """
        # Restore observations and clear all derived state.
        self._flux = _readonly(self.original_flux)
        self._flux_err = (
            None if self.original_flux_err is None else _readonly(self.original_flux_err)
        )
        self.history = []
        self.prewhitening = None
        self.truth_mask[:] = False
        self._invalidate()
        return self

    def standardize(self):
        """Subtract the mean from working flux and divide flux and errors by its standard deviation.

        Uses the population standard deviation (ddof=0). Rejects constant or numerically
        indistinguishable-from-constant flux. Originals are preserved; fits and searches
        are invalidated. Returns this object.

        :returns: This object, for chaining.
        :rtype: Nahiku
        """
        # Measure the working-data offset and scale before transforming flux and errors.
        offset, scale = float(np.mean(self.flux)), float(np.std(self.flux))
        if scale <= np.finfo(float).eps * max(abs(offset), 1):
            raise ValueError("Cannot standardize constant flux")
        self._flux = _readonly((self.flux - offset) / scale)
        if self.flux_err is not None:
            self._flux_err = _readonly(self.flux_err / scale)
        # Record the transformation and invalidate predictions made in the previous units.
        self.history.append(dict(operation="standardize", offset=offset, scale=scale))
        self._invalidate()
        return self

    def bin(self, width):
        """Return an independently binned Nahiku object; ``width`` is days.

        Use inverse-variance weighted means when errors are available and
        unweighted means otherwise. Propagated errors assume independent noise;
        binning does not model correlations. Empty bins are omitted. Metadata
        records source sample groups; the parent and its results are unchanged.
        At least three occupied bins must remain. The returned object uses the binned
        working values as its originals, so resetting it does not restore unbinned data.
        A bin inherits an injection label if any contributing sample is labeled.

        :param width: Bin width in days; must be finite and positive.
        :type width: float
        :returns: Independent binned object with propagated errors and sample lineage.
        :rtype: Nahiku
        """
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Bin width must be finite and positive")
        # Assign samples to fixed-width bins anchored at the first timestamp.
        groups = np.floor((self.time - self.time[0]) / width).astype(int)
        rows, lineage, truth = [], [], []
        # Average each occupied bin and propagate independent measurement errors.
        for group in np.unique(groups):
            indices = np.flatnonzero(groups == group)
            weights = (
                np.ones(len(indices)) if self.flux_err is None else self.flux_err[indices] ** -2
            )
            rows.append(
                [
                    np.average(self.time[indices], weights=weights),
                    np.average(self.flux[indices], weights=weights),
                    1 / np.sqrt(weights.sum()),
                ]
            )
            lineage.append(indices.tolist())
            truth.append(bool(self.truth_mask[indices].any()))
        # Build an independent light curve with source groups and inherited truth labels.
        values = np.array(rows)
        meta = deepcopy(self.metadata)
        meta["bin_source_groups"] = lineage
        result = Nahiku(
            values[:, 0], values[:, 1], None if self.flux_err is None else values[:, 2], meta
        )
        result.history = deepcopy(self.history) + [dict(operation="bin", width=width)]
        result.truth_mask = np.array(truth)
        return result

    def estimate_period(
        self, *, minimum_period=None, maximum_period=None, samples_per_peak=10, maximum_fap=0.01,
        fallback=None,
    ):
        """Estimate a period from observation timestamps using Lomb–Scargle.

        Bounds are days; defaults are twice the median spacing and the baseline.
        Store the spectrum and approximate false-alarm probability in
        ``period_result``. Return a period in days or None when inconclusive.
        Accept the strongest peak only when its false-alarm probability is finite
        and at most ``maximum_fap``, and it lies strictly inside the frequency
        grid; peaks at either grid boundary are rejected.
        The false-alarm probability (FAP) assumes the Lomb–Scargle noise model;
        it is not calibrated for arbitrary correlated noise.
        With ``fallback="shoulder"``, an unaccepted peak triggers Gaussian smoothing
        (sigma=2 frequency bins) and selection of the first negative spectral slope.
        This heuristic timescale is capped at the observation baseline; it is not
        an accepted periodic detection. Diagnostics retain the original peak FAP
        and mark the fallback separately. No usable shoulder returns None.

        :param minimum_period: Shortest tested period in days; None uses twice the median cadence.
        :type minimum_period: float or None
        :param maximum_period: Longest tested period in days; None uses the observation baseline.
        :type maximum_period: float or None
        :param samples_per_peak: Frequency-grid samples per expected periodogram peak;
            at least two, with a default of ten. Larger values refine the frequency grid
            without adding observations.
        :type samples_per_peak: float
        :param maximum_fap: Largest accepted false-alarm probability, in (0, 1].
        :type maximum_fap: float
        :param fallback: None (default) returns no period for an unaccepted peak;
            "shoulder" tries the smoothed-spectrum heuristic for GP initialization.
        :type fallback: str or None
        :returns: Accepted period or requested heuristic timescale in days, or None
            when neither is available within the supplied bounds.
        :rtype: float or None
        """
        if fallback not in (None, "shoulder"):
            raise ValueError("fallback must be None or 'shoulder'")
        # Choose period bounds from explicit settings or the observed sampling.
        lower = minimum_period if minimum_period is not None else 2 * np.median(np.diff(self.time))
        upper = maximum_period if maximum_period is not None else np.ptp(self.time)
        if not 0 < lower < upper or not 0 < maximum_fap <= 1 or samples_per_peak < 2:
            raise ValueError("Invalid period bounds or significance settings")
        if np.std(self.flux) == 0:
            self.period = None
            self.period_result = dict(
                period=None, reason="constant_flux", peak_accepted=False,
                fallback=fallback, fallback_used=False,
            )
            return None
        # Compute the periodogram using actual timestamps and available errors.
        ls = LombScargle(self.time - self.time[0], self.flux, dy=self.flux_err)
        frequency, power = ls.autopower(
            minimum_frequency=1 / upper,
            maximum_frequency=1 / lower,
            samples_per_peak=samples_per_peak,
        )
        index = int(np.argmax(power))
        fap = float(
            ls.false_alarm_probability(
                power[index], minimum_frequency=1 / upper, maximum_frequency=1 / lower
            )
        )
        # Accept only significant interior peaks and retain diagnostics even on rejection.
        accepted = np.isfinite(fap) and fap <= maximum_fap and 0 < index < len(power) - 1
        self.period = float(1 / frequency[index]) if accepted else None
        self.period_result = dict(
            period=self.period,
            candidate_period=float(1 / frequency[index]),
            frequency=frequency,
            power=power,
            false_alarm_probability=fap,
            reason="accepted" if accepted else "inconclusive_or_boundary_peak",
            peak_accepted=bool(accepted),
            fallback=fallback,
            fallback_used=False,
        )
        if not accepted and fallback == "shoulder":
            # Smooth the sampled spectrum and find its first finite descending slope.
            smooth_power = gaussian_filter1d(power, sigma=2)
            slope = (
                np.gradient(smooth_power, frequency)
                if len(frequency) > 1 else np.full_like(frequency, np.nan)
            )
            shoulders = np.flatnonzero(np.isfinite(slope) & (slope < 0) & (frequency > 0))
            self.period_result.update(
                smoothed_power=smooth_power, spectral_slope=slope, reason="no_shoulder"
            )
            if len(shoulders):
                shoulder = int(shoulders[0])
                raw_period = float(1 / frequency[shoulder])
                # Cap the heuristic at the duration, independent of the absolute time origin.
                period = min(raw_period, float(np.ptp(self.time)), float(upper))
                self.period_result.update(
                    shoulder_index=shoulder, shoulder_period_uncapped=raw_period,
                    reason="shoulder_outside_bounds",
                )
                if lower <= period <= upper:
                    self.period = period
                    self.period_result.update(
                        period=period, reason="shoulder_fallback", fallback_used=True
                    )
        return self.period

    def prewhiten(
        self, *, fmin, fmax, max_components=20, minimum_snr=4.0, samples_per_peak=None, fit_mask=None,
        method="balmung",
    ):
        """Remove fitted sinusoids; return and retain a PrewhiteningResult.

        Frequencies are cycles/day. The default delegates to Dan Hey’s Balmung,
        which uses an unweighted joint sinusoid fit
        and a threshold fixed from the initial amplitude spectrum. The weighted
        method refines frequency locally and recalculates the spectral threshold.
        No automatic standardization follows removal; the continuum is preserved.
        For Balmung, frequency bounds constrain peak initialization, not its joint fit.
        Reported errors do not include uncertainty in fitted pulsation parameters.
        Balmung optimizer errors propagate without changing working observations.

        :param fmin: Positive lower frequency bound in cycles per day.
        :type fmin: float
        :param fmax: Upper frequency bound in cycles per day; greater than fmin.
        :type fmax: float
        :param max_components: Maximum number of sinusoidal components to remove.
        :type max_components: int
        :param minimum_snr: Amplitude-to-background cutoff; fixed from the initial spectrum
            for Balmung, recomputed from residual power for the weighted method.
        :type minimum_snr: float
        :param samples_per_peak: None selects Balmung's fixed fivefold frequency sampling or
            ten samples per peak for the weighted method. Balmung accepts only None or 5;
            the weighted method accepts integers of at least two.
        :type samples_per_peak: int or None
        :param fit_mask: Boolean mask selecting samples used to fit sinusoids; None uses all
            samples. The fitted signal is subtracted at all times.
        :type fit_mask: array-like or None
        :param method: "balmung" (default) or "weighted". Balmung does not weight by
            supplied flux errors and rejects fit_mask; "weighted" supports both.
            Errors remain unchanged by either method.
        :type method: str
        :returns: Removed components, residuals, and before/after spectra.
        :rtype: PrewhiteningResult
        """
        # Fit and remove pulsations using the selected frequency range and protected samples.
        result = _prewhiten(
            self.time,
            self.flux,
            self.flux_err,
            fmin=fmin,
            fmax=fmax,
            max_components=max_components,
            minimum_snr=minimum_snr,
            samples_per_peak=samples_per_peak,
            fit_mask=fit_mask,
            method=method,
        )
        # Replace working flux while retaining component diagnostics and processing history.
        self._flux = _readonly(result.residuals)
        self.prewhitening = result
        self.history.append(
            dict(
                operation="prewhiten",
                method=method,
                samples_per_peak=(5 if method == "balmung" else 10) if samples_per_peak is None else samples_per_peak,
                fmin=fmin,
                fmax=fmax,
                max_components=max_components,
                minimum_snr=minimum_snr,
                frequencies=result.frequencies.tolist(),
                stopping_reason=result.stopping_reason,
                protected_samples=0
                if fit_mask is None
                else int((~np.asarray(fit_mask, bool)).sum()),
            )
        )
        self._invalidate()
        return result

    def fit_gp(
        self,
        *,
        period=None,
        kernel="quasiperiodic",
        length_scale=None,
        noise=None,
        iterations=1000,
        learning_rate=0.01,
        optimizer="adam",
        patience=1,
        device="cpu",
        mean=None,
        seed=48,
        qp_constraints=None,
        standardize=True,
        objective="mll",
        early_stopping=True,
        min_iterations=None,
        save_state="last",
        set_noise_equal_to_var_residuals=False,
        dtype="float64",
        return_mll=False,

    ):
        """Fit an exact GP and return this object.

        ``period`` and ``length_scale`` are days; noise is an additional flux standard deviation.
        ``kernel`` accepts 'quasiperiodic', 'rbf', or a GPyTorch kernel; ``mean``
        accepts a GPyTorch mean. Known measurement variance is added to learned
        additional noise variance; numerical Cholesky jitter is a separate adjustment. ``iterations``, ``learning_rate``, ``optimizer`` ('adam'/'sgd')
        and ``patience`` control optimization. CPU/float64 is the default.
        ``seed`` is recorded in provenance but does not set a random seed. Built-in
        initialization is deterministic; GPyTorch may use iterative or stochastic
        calculations for training matrices at or above 10,000 samples.

        :param period: Initial recurrence period in days; None uses the stored period estimate,
            including a requested shoulder fallback. Used only by the built-in QP kernel.
        :type period: float or None
        :param kernel: Covariance specification: "quasiperiodic", "rbf", or a custom kernel copied
            before fitting.
        :type kernel: str or gpytorch.kernels.Kernel
        :param length_scale: Initial RBF timescale in days; None uses 1 day without a preset.
            Ignored for custom kernels; initialize their parameters before passing them.
            Cannot be supplied with period-informed constraints.
        :type length_scale: float or None
        :param noise: Initial additional noise standard deviation in flux units; None estimates it
            from adjacent differences without a preset. With measurement errors, the initial
            standard deviation is max(0.1 * std(diff(flux)), 1e-5); without errors it is
            max(std(diff(flux)) / sqrt(2), 1e-5). The period-informed preset starts
            the additional noise variance at 0.5.
        :type noise: float or None
        :param iterations: Positive maximum number of optimization iterations.
        :type iterations: int
        :param learning_rate: Positive optimizer step size.
        :type learning_rate: float
        :param optimizer: Optimizer name: "adam" or "sgd".
        :type optimizer: str
        :param patience: Stop when consecutive non-improvements exceed this count after warm-up.
            Any strictly smaller loss counts as an improvement.
        :type patience: int
        :param device: Computation device; CPU is the default and CUDA requires an available GPU.
        :type device: str or torch.device
        :param mean: Custom mean module, copied before fitting; None initializes a constant mean
            from median flux, or 0.01 with the period-informed preset.
        :type mean: gpytorch.means.Mean or None
        :param seed: Seed recorded in provenance; it does not seed PyTorch or custom modules.
            Built-in initialization is deterministic; large-matrix GPyTorch calculations
            or user-supplied modules may use randomness.
        :type seed: int
        :param qp_constraints: None or "period-informed" to apply period-informed QP,
            mean, outputscale, and noise constraints. No length_scale, noise, or mean overrides.
        :type qp_constraints: str or None
        :param standardize: Standardize working flux and errors unless mean is within 1e-8
            of zero and population standard deviation is within 1e-6 of one.
            False preserves the supplied units. Explicit noise and mean settings must
            use the resulting working units. Originals are always preserved.
        :type standardize: bool
        :param objective: "mll" minimizes negative marginal log likelihood; "mse" compares
            the training-mode prior mean to flux and does not train kernel/noise parameters.
        :type objective: str
        :param early_stopping: Enable stopping after the warm-up and patience checks.
        :type early_stopping: bool
        :param min_iterations: Zero-based warm-up boundary; checks start when iteration index
            exceeds it. None uses iterations // 10.
        :type min_iterations: int or None
        :param save_state: "last" retains the final model parameters; "best" restores the
            model state with the lowest evaluated training objective, including warm-up evaluations.
            Optimizer momentum and other optimizer state are not saved.
        :type save_state: str
        :param set_noise_equal_to_var_residuals: Reset learned noise variance after state selection
            to max(1e-4, training posterior residual variance). Known error variances remain
            additional fixed terms. Residuals are measured before the reset; no reoptimization.
        :type set_noise_equal_to_var_residuals: bool
        :param dtype: "float32" or "float64" for training and dense inference arrays.
        :type dtype: str
        :param return_mll: Return (self, marginal_log_likelihood) instead of self.
            The likelihood is per observation, evaluated after any noise reset.
        :type return_mll: bool
        :returns: This object, or (this object, final marginal log likelihood) when requested.
        :rtype: Nahiku or tuple
        """
        if not isinstance(return_mll, bool):
            raise ValueError("return_mll must be True or False")
        if not isinstance(standardize, bool):
            raise ValueError("standardize must be True or False")
        # Resolve the period and collect the exact GP configuration for later subset refits.
        options = dict(
            period=self.period if period is None else period,
            kernel=kernel,
            length_scale=length_scale,
            noise=noise,
            iterations=iterations,
            learning_rate=learning_rate,
            optimizer=optimizer,
            patience=patience,
            device=device,
            mean=mean,
            seed=seed,
            qp_constraints=qp_constraints,
            objective=objective, early_stopping=early_stopping, min_iterations=min_iterations,
            save_state=save_state, set_noise_equal_to_var_residuals=set_noise_equal_to_var_residuals,
            dtype=dtype,
        )
        # Replace fitted state only after optimization succeeds, then clear prior searches.
        offset, scale = float(np.mean(self.flux)), float(np.std(self.flux))
        needs_scaling = standardize and not (
            abs(offset) <= 1e-8 and abs(scale - 1) <= 1e-6
        )
        if needs_scaling and scale <= np.finfo(float).eps * max(abs(offset), 1):
            raise ValueError("Cannot standardize constant flux; use standardize=False to fit as supplied")
        # Fit prepared arrays first, so a failed fit leaves existing observations untouched.
        flux = (self.flux - offset) / scale if needs_scaling else self.flux
        errors = self.flux_err
        if needs_scaling and errors is not None:
            errors = errors / scale
        fit = GPFit(self.time, flux, errors, **options)
        if needs_scaling:
            saved_period, saved_result = self.period, self.period_result
            self.standardize()
            # An affine flux transformation does not change the estimated recurrence time.
            self.period, self.period_result = saved_period, saved_result
        fit.settings["standardization"] = dict(
            requested=standardize, applied=needs_scaling,
            offset=offset if needs_scaling else 0.0, scale=scale if needs_scaling else 1.0,
        )
        self._fit, self._fit_options = fit, deepcopy(options)
        self.results = {}
        return (self, fit.marginal_log_likelihood) if return_mll else self

    def plot_fit(self, axes=None):
        """Plot the current GP optimization loss and covariance matrices.

        :param axes: Two axes, or four for a built-in QP kernel; None creates a figure.
        :type axes: sequence of matplotlib.axes.Axes or None
        :returns: Figure and axes; call show() explicitly to display them.
        :rtype: tuple
        """
        return self._require_fit().plot(axes=axes)

    def _require_fit(self):
        """Require a fit made after the latest working-data transformation.

        :returns: Current fitted GP; raises RuntimeError if no valid fit exists.
        :rtype: GPFit
        """
        if self._fit is None:
            raise RuntimeError("Call fit_gp() after the most recent data transformation")
        return self._fit

    def _refit_subset(self, indices, initial_fit=None):
        """Fit the selected observations using the current GP configuration.

        :param indices: Integer sample indices selecting the requested subset of the full
            observation grid.
        :type indices: array-like
        :param initial_fit: Previous GP supplying learned parameters, or None for fresh initialization.
        :type initial_fit: GPFit or None
        :returns: Independent GP fitted to the selected observations.
        :rtype: GPFit
        """
        return GPFit(
            self.time[indices],
            self.flux[indices],
            None if self.flux_err is None else self.flux_err[indices],
            initial_fit=initial_fit,
            **self._fit_options,
        )

    def _full_prior(self, fit):
        """Evaluate a fitted prior and observation covariance on the full grid.

        :param fit: Fitted GP whose mean, covariance, and noise are evaluated on the full
            observation grid.
        :type fit: GPFit
        :returns: Full noisy covariance matrix and prior mean vector.
        :rtype: tuple
        """
        import torch

        # Evaluate the fitted mean and kernel at every original observation time.
        x = torch.as_tensor(self.time - fit.origin, dtype=fit.torch_dtype, device=fit.device)
        with torch.no_grad():
            mean = fit.model.mean_module(x).cpu().numpy()
            covariance = fit.model.covar_module(x).to_dense().cpu().numpy()
        # Add learned additional noise and measurement variance to the observation covariance.
        noise = np.full(len(x), fit.noise_variance, dtype=fit.numpy_dtype)
        if self.flux_err is not None:
            noise += self.flux_err**2
        return covariance + np.diag(noise), mean

    def predict(self, time=None, *, include_noise=False, flux_err=None):
        """Return the GP mean and standard deviation at timestamps in days.

        Default uncertainty describes the latent function. ``include_noise=True``
        adds learned additional noise variance and supplied measurement-error variance.
        Stored errors are used automatically only when ``time=None``. Supplying
        ``time=lc.time`` explicitly requires ``flux_err=lc.flux_err`` to include them.

        :param time: Finite prediction timestamps in days; None uses observed timestamps and stored
            errors.
        :type time: array-like or None
        :param include_noise: Include learned additional noise and supplied measurement variance in predictive
            uncertainty.
        :type include_noise: bool
        :param flux_err: Positive prediction errors matching time; added only with
            include_noise=True, but validated whenever supplied. When time=None, stored errors take precedence.
        :type flux_err: array-like or None
        :returns: Predictive mean and pointwise standard deviation in working flux units.
        :rtype: tuple
        """
        fit = self._require_fit()
        # Use stored observations and uncertainties when no prediction grid is supplied.
        if time is None:
            time, flux_err = self.time, self.flux_err
        # Validate the prediction grid and any supplied measurement errors.
        time = np.asarray(time, float)
        if time.ndim != 1 or not np.isfinite(time).all():
            raise ValueError("Prediction time must be a finite 1D array")
        if flux_err is not None:
            flux_err = np.asarray(flux_err, float)
            if (
                flux_err.shape != time.shape
                or np.any(flux_err <= 0)
                or not np.isfinite(flux_err).all()
            ):
                raise ValueError("Prediction errors must be finite positive values matching time")
        return fit.predict(time, include_noise=include_noise, errors=flux_err)

    def greedy_search(
        self, *, sigma=3.0, grow_sigma=2.0, max_events=None, direction="both",
        refit=False, gap_factor=None, backend="torch", seed_statistic="residual",
        len_deviant=1, growth="metric", growth_metric="mll", expansion_param=1,
        warm_initialization=True, update_threshold=False, min_remaining=3,
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
        from .greedy_search import greedy_search

        result = greedy_search(
            self, sigma=sigma, grow_sigma=grow_sigma, max_events=max_events,
            direction=direction, refit=refit, gap_factor=gap_factor, backend=backend,
            seed_statistic=seed_statistic, len_deviant=len_deviant, growth=growth,
            growth_metric=growth_metric, expansion_param=expansion_param,
            warm_initialization=warm_initialization, update_threshold=update_threshold,
            min_remaining=min_remaining,
        )
        # Store this search independently of any exhaustive-search result.
        self.results["greedy"] = result
        return result

    def exhaustive_search(
        self,
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
        from .exhaustive_search import exhaustive_search

        result = exhaustive_search(
            self,
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
            backend=backend,
            warm_initialization=warm_initialization,
        )
        # Store this search independently of any greedy-search result.
        self.results["exhaustive"] = result
        return result

    def inject_anomaly(self, *, center, width, depth, shape="gaussian", alpha=3.0):
        """Inject a negative profile in working units; return its flux array.

        ``center`` and ``width`` are days, ``depth`` is positive flux. Width is
        Gaussian sigma, saw half-width, or exocomet ingress sigma. Exocomet
        egress is exponential with timescale ``alpha*width``. Truth support uses
        1% of peak depth; profiles are truncated at that support.

        The exocomet shape is the Gaussian–exponential empirical template in
        `Kennedy et al. (2019), An automated search for transiting exocomets,
        equation 6 <https://doi.org/10.1093/mnras/sty3049>`_. Before truncation,
        the additive signal is ``-depth*exp(-0.5*u**2)`` for ``u < 0`` and
        ``-depth*exp(-u/alpha)`` otherwise, where ``u = (time-center)/width``.
        The paper's parameters map to ``t0 = center``, ``theta = width``, and
        ``lambda = alpha*width``; its fractional depth A corresponds to ``depth``
        when the input continuum is unity. Here depth is in working flux units.
        Our ``alpha`` is a timescale ratio, not the paper's fit-comparison
        asymmetry statistic. The center is the continuous profile's deepest
        point; discrete observations need not sample that point exactly.

        This template is useful for injection/recovery tests of asymmetric dips;
        it is not a physical dust-tail model or evidence that a detected dip is
        cometary. The 1%-depth truncation is a numerical convention in this
        implementation, not part of the cited equation, and introduces small
        discontinuities at the support boundaries.

        :param center: Dip center in days, within the observed time span.
        :type center: float
        :param width: Positive width in days: Gaussian sigma, saw half-width, or exocomet ingress
            sigma.
        :type width: float
        :param depth: Positive peak depth in current working flux units.
        :type depth: float
        :param shape: Injection profile: "gaussian", "saw", or "exocomet".
        :type shape: str
        :param alpha: Positive exocomet egress timescale divided by ingress width.
        :type alpha: float
        :returns: Injected negative signal on the observation grid.
        :rtype: numpy.ndarray
        """
        if shape not in ("gaussian", "saw", "exocomet") or width <= 0 or depth <= 0 or alpha <= 0:
            raise ValueError("Provide a supported shape and positive width, depth, alpha")
        if (
            not np.isfinite([center, width, depth, alpha]).all()
            or not self.time[0] <= center <= self.time[-1]
        ):
            raise ValueError("Injection parameters must be finite and center must lie within data")
        # Evaluate the requested dip profile in dimensionless time from its center.
        u = (self.time - center) / width
        if shape == "gaussian":
            profile = np.exp(-0.5 * u * u)
        elif shape == "saw":
            profile = np.maximum(1 - np.abs(u), 0)
        else:
            profile = np.where(u < 0, np.exp(-0.5 * u * u), np.exp(-np.maximum(u, 0) / alpha))
        # Truncate the profile at one percent of its depth and mark injection truth.
        support = profile >= 0.01
        signal = -depth * np.where(support, profile, 0)
        self._flux = _readonly(self.flux + signal)
        self.truth_mask |= support
        self.history.append(
            dict(
                operation="inject_anomaly",
                center=center,
                width=width,
                depth=depth,
                shape=shape,
                alpha=alpha,
            )
        )
        self._invalidate()
        return signal

    def evaluate(self, result, *, truth=None, buffer=0):
        """Return one-to-one event precision/recall against injected or supplied truth.

        ``truth`` is a boolean sample mask; ``buffer`` expands overlap tolerance
        by that many samples. Real-star detections have no implied truth labels.

        :param result: Search result evaluated on this observation grid.
        :type result: SearchResult
        :param truth: Boolean sample mask of true events; None uses accumulated injection labels.
        :type truth: array-like or None
        :param buffer: Nonnegative overlap tolerance in samples, applied to the truth intervals.
        :type buffer: int
        :returns: Matched-event counts and one-to-one event precision and recall.
        :rtype: dict
        """
        # Use injected truth by default and require matching observation grids.
        truth = self.truth_mask if truth is None else np.asarray(truth, bool)
        if truth.shape != self.time.shape or not np.array_equal(result.time, self.time):
            raise ValueError("Result and truth must refer to this observation grid")
        return evaluate_events(result.intervals, events_from_mask(truth), buffer)

    def plot(self, ax=None, *, original=False):
        """Return ``(figure, ax)`` showing working or original flux, without display.

        :param ax: Axes to draw on; None creates a new figure and axes.
        :type ax: matplotlib.axes.Axes or None
        :param original: Plot original observations instead of the current working flux.
        :type original: bool
        :returns: Figure and axes handles; no display is triggered.
        :rtype: tuple
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 3))
        ax.plot(self.time, self.original_flux if original else self.flux, ".", ms=2)
        ax.set(
            xlabel=f"Time [days; {self.metadata.get('time_format', 'input reference')}]",
            ylabel="Original flux" if original else "Working flux",
        )
        return ax.figure, ax

    def export(self, path):
        """Export working/original arrays to NPZ and processing provenance to JSON.

        Missing error arrays are stored as empty arrays. The NPZ includes the truth
        mask; JSON includes processing history and fitted settings when available.
        This export does not include SearchResult arrays or complete prewhitening
        spectra; export search results separately and save prewhitening arrays as needed.
        Existing files with the same prefix are overwritten.

        :param path: Output filename prefix; required suffixes are appended and parent directories
            are created.
        :type path: str or pathlib.Path
        :returns: None.
        :rtype: None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save numerical arrays without serializing a live model.
        np.savez_compressed(
            str(path) + ".npz",
            time=self.time,
            flux=self.flux,
            original_flux=self.original_flux,
            flux_err=np.array([]) if self.flux_err is None else self.flux_err,
            original_flux_err=np.array([])
            if self.original_flux_err is None
            else self.original_flux_err,
            truth_mask=self.truth_mask,
        )
        # Write processing and fit provenance alongside the arrays.
        Path(str(path) + ".json").write_text(
            json.dumps(self._provenance(), indent=2, allow_nan=False)
        )

    @classmethod
    def from_lightkurve(cls, lightcurve, *, normalize=True):
        """Convert an existing Lightkurve object; filter invalid samples and sort.

        Preserve available positive uncertainties, time reference, original row indices, and
        selected observation metadata. If any finite positive errors are present,
        samples with invalid errors are removed. If no such errors exist, errors
        are treated as unavailable and finite time/flux samples are retained.
        At least three samples and strictly increasing times must remain.
        Time values are copied without conversion; supply a time format measured in days.
        Normalization divides by the retained median flux;
        no archive query or quality filtering occurs here (apply quality masks first).

        :param lightcurve: Existing light curve with time, flux, flux_err, and metadata; apply
            quality masks before conversion.
        :type lightcurve: lightkurve.LightCurve
        :param normalize: Divide flux and errors by median flux and its absolute value,
            respectively.
        :type normalize: bool
        :returns: Independent new light-curve object.
        :rtype: Nahiku
        """
        t = np.asarray(lightcurve.time.value, float)
        y = np.asarray(lightcurve.flux.value, float)
        e = np.asarray(lightcurve.flux_err.value, float)
        # Retain finite samples and, when available, finite positive uncertainties.
        has_errors = bool(np.any(np.isfinite(e) & (e > 0)))
        keep = np.isfinite(t) & np.isfinite(y)
        if has_errors:
            keep &= np.isfinite(e) & (e > 0)
        # Sort retained rows while preserving their original indices for provenance.
        indices = np.flatnonzero(keep)
        indices = indices[np.argsort(t[indices], kind="stable")]
        # Normalize flux and uncertainties together using the retained median flux.
        scale = float(np.median(y[indices])) if normalize else 1.0
        if scale == 0 or not np.isfinite(scale):
            raise ValueError("Cannot normalize zero/invalid median flux")
        # Preserve the time reference, flux units, and source-row mapping.
        meta = dict(
            time_format=lightcurve.time.format,
            time_scale=lightcurve.time.scale,
            source_indices=indices.tolist(),
            normalization=scale,
            flux_unit=str(lightcurve.flux.unit),
        )
        for key in ("TARGETID", "LABEL", "SECTOR", "QUARTER", "MISSION", "AUTHOR"):
            if key in lightcurve.meta:
                meta[key.lower()] = str(lightcurve.meta[key])
        return cls(
            t[indices], y[indices] / scale, e[indices] / abs(scale) if has_errors else None, meta
        )

    @classmethod
    def download(
        cls,
        target,
        *,
        mission,
        author,
        exptime,
        quarter=None,
        sector=None,
        cache_dir=None,
        quality_bitmask="default",
        stitch=False,
    ):
        """Download one archive product, or explicitly stitch all matching products.

        Specify mission, pipeline author, cadence (seconds), and quarter/sector.
        With ``stitch=False`` (default), require exactly one matching product.
        With ``stitch=True``, download all matches and use Lightkurve's stitching,
        which median-normalizes each segment before concatenation. Quarter/sector
        may be a single value or a list; omitting them allows all matching epochs.
        ``cache_dir`` controls downloaded FITS storage, and ``quality_bitmask``
        applies to every downloaded product. Empty or incomplete downloads raise an error.
        Requires the ``data`` extra. Return a normalized Nahiku object.

        Conversion through :meth:`from_lightkurve` filters non-finite time/flux
        samples, sorts by time, and handles available positive uncertainties.
        Duplicate timestamps are rejected; select non-overlapping products.
        Stitching preserves observing gaps and does not flatten or detrend data.
        Metadata records each input product and its concatenated row range;
        ``source_indices`` maps retained samples into that concatenation.

        :param target: Target identifier or sky position accepted by lightkurve.search_lightcurve.
        :type target: str or object
        :param mission: Mission selection passed to Lightkurve, such as "Kepler" or "TESS".
        :type mission: str or sequence of str
        :param author: Pipeline selection passed to Lightkurve, such as "Kepler" or "SPOC".
        :type author: str or sequence of str
        :param exptime: Cadence in seconds or a cadence label accepted by Lightkurve.
        :type exptime: float or str
        :param quarter: Kepler quarter selection; None allows all matching quarters.
        :type quarter: int or sequence of int or None
        :param sector: TESS sector selection; None allows all matching sectors.
        :type sector: int or sequence of int or None
        :param cache_dir: FITS download directory; None uses the default Lightkurve cache.
        :type cache_dir: str or None
        :param quality_bitmask: Quality-flag mask passed to every Lightkurve download.
        :type quality_bitmask: str or int
        :param stitch: Download and median-normalize all matching products before concatenation;
            False requires exactly one match.
        :type stitch: bool
        :returns: Independent new light-curve object.
        :rtype: Nahiku

        See Also
        --------
        `lightkurve.search_lightcurve documentation <https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html>`_
            Archive search parameters and supported target identifiers. This
            wrapper exposes the selection arguments listed in its signature.
        """
        import lightkurve as lk

        if not isinstance(stitch, bool):
            raise ValueError("stitch must be True or False")
        # Build an explicit archive selection and require the requested product count.
        selection = dict(mission=mission, author=author, exptime=exptime)
        if quarter is not None:
            selection["quarter"] = quarter
        if sector is not None:
            selection["sector"] = sector
        found = lk.search_lightcurve(target, **selection)
        if len(found) == 0:
            raise ValueError("No products found; refine selection")
        if not stitch and len(found) != 1:
            raise ValueError(
                f"Expected one product, found {len(found)}; refine selection or set stitch=True"
            )
        # Download one product or all matches, rejecting missing or partial responses.
        options = dict(download_dir=cache_dir, quality_bitmask=quality_bitmask)
        if stitch:
            products = found.download_all(**options)
            if products is None or len(products) != len(found) or any(p is None for p in products):
                raise RuntimeError("Archive download did not return every selected light curve")
        else:
            lc = found.download(**options)
            if lc is None:
                raise RuntimeError("Archive download returned no light curve")
            products = [lc]
        # Record each product and its row range before stitching and filtering.
        provenance, offset = [], 0
        for product in products:
            record = dict(start=offset, stop=offset + len(product))
            for key in ("TARGETID", "LABEL", "SECTOR", "QUARTER", "MISSION", "AUTHOR", "FILENAME"):
                if key in product.meta:
                    record[key.lower()] = str(product.meta[key])
            provenance.append(record)
            offset += len(product)
        if stitch:
            # Median-normalize segments with Lightkurve before converting the combined series.
            lc = products.stitch()
        result = cls.from_lightkurve(lc)
        if len(products) > 1:
            # A stitched curve may inherit its first segment's metadata.
            result.metadata.pop("sector", None)
            result.metadata.pop("quarter", None)
        result.metadata.update(
            target=str(target), selection=selection, quality_bitmask=quality_bitmask,
            stitched=stitch, product_count=len(products), source_products=provenance,
        )
        return result

    @classmethod
    def from_synthetic_noise(
        cls,
        *,
        duration=10.0,
        n_points=300,
        period=2.0,
        amplitude=0.3,
        white_noise=0.05,
        red_noise=0.03,
        red_timescale=0.3,
        slope=0.0,
        phase=0.0,
        seed=48,
    ):
        """Generate sinusoidal variability, Gaussian white/OU red noise, and a trend.

        Durations, period, and red_timescale are days; amplitudes are flux units;
        slope is flux/day. ``seed`` controls a local random number generator. Anomalies require
        a separate injection call.
        The returned object's ``flux_err`` contains only the white-noise standard
        deviation; it does not include correlated red noise. When ``white_noise=0``,
        ``flux_err`` is None.

        :param duration: Positive simulation duration in days, including both endpoints.
        :type duration: float
        :param n_points: Number of uniformly spaced simulated observations; at least three.
        :type n_points: int
        :param period: Positive recurrence period in days.
        :type period: float
        :param amplitude: Sinusoid amplitude in flux units.
        :type amplitude: float
        :param white_noise: Nonnegative independent Gaussian noise standard deviation in flux units.
        :type white_noise: float
        :param red_noise: Nonnegative stationary Ornstein–Uhlenbeck noise standard deviation in flux
            units.
        :type red_noise: float
        :param red_timescale: Positive red-noise correlation timescale in days.
        :type red_timescale: float
        :param slope: Linear trend slope in flux units per day, centered on the simulation midpoint.
        :type slope: float
        :param phase: Cosine phase at time zero in radians.
        :type phase: float
        :param seed: Seed for a local NumPy random generator; use an integer for reproducibility.
        :type seed: int or None
        :returns: Independent new light-curve object.
        :rtype: Nahiku
        """
        if (
            duration <= 0
            or n_points < 3
            or period <= 0
            or red_timescale <= 0
            or min(white_noise, red_noise) < 0
        ):
            raise ValueError("Invalid simulation parameters")
        # Use a local random generator so simulation does not alter global random state.
        rng = np.random.default_rng(seed)
        t = np.linspace(0, duration, n_points)
        # Generate stationary correlated noise with the Ornstein–Uhlenbeck recurrence.
        red = np.zeros(n_points)
        red[0] = rng.normal(0, red_noise)
        decay = np.exp(-(t[1] - t[0]) / red_timescale)
        for i in range(1, n_points):
            red[i] = decay * red[i - 1] + red_noise * np.sqrt(1 - decay**2) * rng.normal()
        # Combine periodic variability, a centered trend, and red and white noise.
        y = (
            amplitude * np.cos(2 * np.pi * t / period + phase)
            + slope * (t - duration / 2)
            + red
            + rng.normal(0, white_noise, n_points)
        )
        errors = np.full(n_points, white_noise) if white_noise > 0 else None
        return cls(t, y, errors, dict(generator="noise", seed=seed, time_format="relative"))

    @classmethod
    def from_synthetic_gp(
        cls,
        *,
        duration=10.0,
        n_points=200,
        period=2.0,
        amplitude=0.3,
        periodic_scale=1.0,
        length_scale=3.0,
        white_noise=0.05,
        mean=0.0,
        seed=48,
    ):
        """Draw a quasi-periodic GP plus Gaussian measurement noise using a local random number generator.

        Time parameters are days; amplitude and white_noise are flux standard deviations. The kernel is
        ``amplitude² exp(-2 sin²(pi dt/period)/periodic_scale - dt²/(2 length_scale²))``,
        matching GPyTorch's periodic-kernel convention. Sampling adds a fixed diagonal
        variance of 1e-12 for Cholesky stability before drawing the GP. ``flux_err``
        records only ``white_noise`` and is None when ``white_noise=0``; it does not
        include that numerical adjustment or correlated GP variability.

        :param duration: Positive simulation duration in days, including both endpoints.
        :type duration: float
        :param n_points: Number of uniformly spaced simulated observations; at least three.
        :type n_points: int
        :param period: Positive recurrence period in days.
        :type period: float
        :param amplitude: Positive GP marginal standard deviation in flux units.
        :type amplitude: float
        :param periodic_scale: Positive dimensionless denominator of the periodic term in the kernel
            exponent.
        :type periodic_scale: float
        :param length_scale: Positive covariance evolution timescale in days.
        :type length_scale: float
        :param white_noise: Nonnegative independent Gaussian noise standard deviation in flux units.
        :type white_noise: float
        :param mean: Constant mean flux.
        :type mean: float
        :param seed: Seed for a local NumPy random generator; use an integer for reproducibility.
        :type seed: int or None
        :returns: Independent new light-curve object.
        :rtype: Nahiku
        """
        if (
            min(duration, period, amplitude, periodic_scale, length_scale) <= 0
            or n_points < 3
            or white_noise < 0
        ):
            raise ValueError("Invalid GP simulation parameters")
        # Use a local random generator so simulation does not alter global random state.
        rng = np.random.default_rng(seed)
        t = np.linspace(0, duration, n_points)
        # Construct the quasi-periodic covariance from all pairwise time separations.
        delta = t[:, None] - t[None, :]
        cov = amplitude**2 * np.exp(
            -2 * np.sin(np.pi * delta / period) ** 2 / periodic_scale
            - 0.5 * (delta / length_scale) ** 2
        )
        # Draw correlated flux with a stabilized Cholesky factor, then add white noise.
        y = mean + np.linalg.cholesky(cov + np.eye(n_points) * 1e-12) @ rng.normal(size=n_points)
        y += rng.normal(0, white_noise, n_points)
        return cls(
            t,
            y,
            np.full(n_points, white_noise) if white_noise > 0 else None,
            dict(generator="quasiperiodic_gp", seed=seed, time_format="relative"),
        )
