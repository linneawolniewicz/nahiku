"""GPyTorch models and exact Gaussian conditioning with selectable precision."""

import copy
import numpy as np
import torch
import gpytorch
from scipy.linalg import cho_solve
from .numerics import stable_cholesky, numpy_factor, CovarianceError, record_recovery


def period_informed_settings(period):
    """Return period-dependent parameter bounds and starting values.

    :param period: Positive dominant period in days.
    :type period: float
    :returns: Constraint specifications as lower bound, upper bound, and initial value.
    :rtype: dict
    """
    if period is None or not np.isfinite(period) or period <= 0:
        raise ValueError("The period-informed preset requires a positive finite period")
    # Select period-dependent bounds, with P=0.5 included in the first interval.
    p = float(period)
    if p <= .5:
        recurrence, periodic, evolution = (max(.001, p-.1), 2., p), (p/4, None, p), (p/4, None, 4*p)
    elif p < 1:
        recurrence, periodic, evolution = (.4, 5., p), (.1, None, p), (1., None, 3*p)
    elif p < 4:
        recurrence, periodic, evolution = (.4, 5., p), (.2, None, p/2), (1., None, 2*p)
    elif p < 8:
        recurrence, periodic, evolution = (p-1, p+1, p), (.4, None, p/4), (p/3, None, 1.5*p)
    else:
        recurrence, periodic, evolution = (p-2, p+2, p), (.4, None, 2.), (p/4, None, p)
    settings = dict(period=recurrence, periodic_lengthscale=periodic, rbf_lengthscale=evolution,
                    outputscale=(0., 5., 1.), mean=(-1., 1., .01), noise=(1e-4, None, .5))
    # Require each initial value to lie strictly inside its parameter bounds.
    for name, (lower, upper, initial) in settings.items():
        if initial <= lower or (upper is not None and initial >= upper):
            raise ValueError(f"Period-informed {name} initialization lies outside its bounds")
    return settings


def _constraint(spec):
    """Construct a GPyTorch constraint from a recorded specification.

    :param spec: Lower bound, optional upper bound, and initial value.
    :type spec: tuple
    :returns: Constraint retaining the requested initial value.
    :rtype: gpytorch.constraints.Interval
    """
    lower, upper, initial = spec
    if upper is None:
        return gpytorch.constraints.GreaterThan(lower, initial_value=initial)
    return gpytorch.constraints.Interval(lower, upper, initial_value=initial)


class QuasiPeriodicKernel(gpytorch.kernels.ProductKernel):
    """Product of GPyTorch periodic and squared-exponential covariance kernels.

    The covariance is exp(-2 sin²(pi dt / P) / l_periodic - dt² / (2 l_rbf²)).
    P and l_rbf use the input time units; l_periodic is dimensionless. This class
    has no amplitude parameter; wrap it in a GPyTorch ScaleKernel to fit one."""

    def __init__(self):
        """Multiply periodic and squared-exponential kernels.

        :returns: None.
        :rtype: None
        """
        super().__init__(gpytorch.kernels.PeriodicKernel(), gpytorch.kernels.RBFKernel())


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP with configurable covariance and mean modules."""

    def __init__(self, x, y, likelihood, kernel, mean):
        """Attach the supplied likelihood, mean, and covariance to an exact GP.

        :param x: Input locations for the GP mean and covariance modules.
        :type x: torch.Tensor
        :param y: Training flux tensor aligned with x.
        :type y: torch.Tensor
        :param likelihood: Observation-noise likelihood passed to the exact GP.
        :type likelihood: gpytorch.likelihoods.Likelihood
        :param kernel: Covariance module assigned to the model.
        :type kernel: gpytorch.kernels.Kernel
        :param mean: Prior mean module assigned to the model.
        :type mean: gpytorch.means.Mean
        :returns: None.
        :rtype: None
        """
        super().__init__(x, y, likelihood)
        self.covar_module = kernel
        self.mean_module = mean

    def forward(self, x):
        """Return the prior multivariate normal at ``x``.

        :param x: Input locations for the GP mean and covariance modules.
        :type x: torch.Tensor
        :returns: Prior multivariate normal distribution at x.
        :rtype: gpytorch.distributions.MultivariateNormal
        """
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def condition(cov, mean, flux, train, test, diagnostics=None):
    """Direct conditional observation distribution, retaining full covariance.

    :param cov: Full observation covariance matrix, including measurement noise, in squared flux
        units.
    :type cov: numpy.ndarray
    :param mean: Prior mean on the full grid.
    :type mean: numpy.ndarray
    :param flux: Flux samples in consistent linear units, aligned with time.
    :type flux: numpy.ndarray
    :param train: Integer indices of conditioning observations.
    :type train: array-like
    :param test: Integer indices at which to return the conditional distribution.
    :type test: array-like
    :param diagnostics: Mutable numerical recovery records; None emits warnings.
    :type diagnostics: list or None
    :returns: Conditional mean and symmetric observation covariance at test indices.
    :rtype: tuple
    """
    train, test = np.asarray(train, int), np.asarray(test, int)
    context = f"direct interval {test.tolist()}"
    # Symmetric cross blocks must describe one consistent covariance.
    asymmetry = float(np.max(np.abs(cov-cov.T)))
    if asymmetry:
        record_recovery(diagnostics, action="symmetrize", context=context, backend="numpy", max_asymmetry=asymmetry)
    cov = (cov + cov.T) / 2
    if len(train):
        cross = cov[np.ix_(test, train)]
        factor = numpy_factor(cov[np.ix_(train, train)], diagnostics, context + " retained")
        prediction = mean[test] + cross @ cho_solve(factor, flux[train] - mean[train])
        covariance = cov[np.ix_(test, test)] - cross @ cho_solve(factor, cross.T)
    else:
        prediction, covariance = mean[test].copy(), cov[np.ix_(test, test)].copy()
    _, covariance = stable_cholesky(covariance, diagnostics, context + " conditional")
    if not np.isfinite(prediction).all():
        raise CovarianceError(context + ": non-finite conditional mean")
    return prediction, covariance


def condition_precision(precision, mean, flux, test, cov=None, diagnostics=None):
    """Conditional observation distribution from a precomputed precision matrix.

    :param precision: Inverse of the full observation covariance matrix.
    :type precision: numpy.ndarray
    :param mean: Prior mean on the full grid.
    :type mean: numpy.ndarray
    :param flux: Flux samples in consistent linear units, aligned with time.
    :type flux: numpy.ndarray
    :param test: Held-out integer indices; all other observations condition the prediction.
    :type test: array-like
    :param cov: Original observation covariance for direct fallback if precision conditioning fails.
    :type cov: numpy.ndarray or None
    :param diagnostics: Mutable numerical recovery records; None emits warnings.
    :type diagnostics: list or None
    :returns: Conditional mean and symmetric observation covariance at held-out indices.
    :rtype: tuple
    """
    test = np.asarray(test, int)
    train = np.setdiff1d(np.arange(len(flux)), test)
    context = f"precision interval {test.tolist()}"
    try:
        # Try the precision identity without jitter before returning to original covariance.
        block = precision[np.ix_(test, test)]
        lower, _ = stable_cholesky(block, diagnostics, context + " block", allow_jitter=False)
        factor = (lower, True)
        covariance = cho_solve(factor, np.eye(len(test), dtype=precision.dtype))
        cross = (precision[np.ix_(test, train)] + precision[np.ix_(train, test)].T) / 2
        prediction = mean[test] - cho_solve(factor, cross @ (flux[train] - mean[train]))
        _, covariance = stable_cholesky(covariance, diagnostics, context + " conditional", allow_jitter=False)
        if not np.isfinite(prediction).all():
            raise CovarianceError(context + ": non-finite conditional mean")
        return prediction, covariance
    except CovarianceError as error:
        if cov is None:
            raise CovarianceError(f"{error} Supply original cov to enable direct conditioning fallback.") from error
        record_recovery(diagnostics, action="direct_fallback", context=context, backend="numpy", reason=str(error))
        return condition(cov, mean, flux, train, test, diagnostics)


class GPFit:
    """Internal fitted state. Dense conditioning costs O(N²) memory and O(N³) setup."""

    def __init__(
        self,
        time,
        flux,
        errors,
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
        objective="mll",
        early_stopping=True,
        min_iterations=None,
        save_state="last",
        set_noise_equal_to_var_residuals=False,
        dtype="float64",
        initial_fit=None,
    ):
        """Optimize an exact GP and cache dense observation-covariance factors.

        :param time: Timestamps in days, on the input time reference.
        :type time: numpy.ndarray
        :param flux: Flux samples in consistent linear units, aligned with time.
        :type flux: numpy.ndarray
        :param errors: Positive one-sigma measurement errors in flux units; None omits known errors.
        :type errors: numpy.ndarray or None
        :param period: Initial recurrence time in days; required for the built-in quasi-periodic kernel.
            This internal class does not estimate a period.
        :type period: float or None
        :param kernel: Covariance specification: "quasiperiodic", "rbf", or a custom kernel copied
            before fitting.
        :type kernel: str or gpytorch.kernels.Kernel
        :param length_scale: Initial RBF timescale in days; None uses 1 day without a preset.
            Ignored for custom kernels; initialize their parameters before passing them.
            Cannot be supplied with the period-informed preset.
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
        :param patience: Stop when consecutive non-improvements exceed this nonnegative count,
            after min_iterations. Any strictly smaller loss counts as an improvement.
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
        :param qp_constraints: None or "period-informed" for period-informed QP,
            mean, outputscale, and noise bounds and initial values. Requires the
            built-in QP kernel and no length_scale, noise, or mean overrides.
        :type qp_constraints: str or None
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
        :param initial_fit: Compatible fitted GP supplying learned parameter values for a warm
            start. Fixed measurement errors come from the new observations; optimizer state is new.
        :type initial_fit: GPFit or None
        :returns: None.
        :rtype: None
        """
        if objective not in ("mll", "mse") or save_state not in ("last", "best"):
            raise ValueError("objective must be 'mll' or 'mse'; save_state must be 'last' or 'best'")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if not isinstance(early_stopping, bool) or not isinstance(set_noise_equal_to_var_residuals, bool):
            raise ValueError("Stopping and noise-reset flags must be bool")
        for name, value, minimum in (("iterations", iterations, 1), ("patience", patience, 0)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        min_iterations = iterations // 10 if min_iterations is None else min_iterations
        if isinstance(min_iterations, bool) or not isinstance(min_iterations, (int, np.integer)) or min_iterations < 0:
            raise ValueError("min_iterations must be a nonnegative integer")
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.numpy_dtype = np.float32 if dtype == "float32" else np.float64
        if qp_constraints not in (None, "period-informed"):
            raise ValueError("qp_constraints must be None or 'period-informed'")
        preset = None
        if qp_constraints is not None:
            if not isinstance(kernel, str) or kernel != "quasiperiodic":
                raise ValueError("Period-informed constraints require kernel='quasiperiodic'")
            if length_scale is not None or noise is not None or mean is not None:
                raise ValueError("Period-informed constraints cannot be combined with length_scale, noise, or mean overrides")
            preset = period_informed_settings(period)
        length_scale = 1.0 if length_scale is None else length_scale
        if iterations < 1 or learning_rate <= 0 or patience < 0 or length_scale <= 0:
            raise ValueError(
                "iterations, learning_rate, patience and length_scale must be positive"
            )
        if optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        # Shift timestamps and use the requested precision on the selected device.
        self.origin = float(time[0])
        self.numerical_diagnostics = []
        self.time = np.array(time)
        self.flux = np.array(flux, dtype=self.numpy_dtype)
        self.device = device
        x = torch.as_tensor(time - self.origin, dtype=self.torch_dtype, device=device)
        y = torch.tensor(flux, dtype=self.torch_dtype, device=device)
        # Initialize a built-in kernel or copy the caller-supplied covariance module.
        if isinstance(kernel, str):
            if kernel == "quasiperiodic":
                if period is None or not np.isfinite(period) or period <= 0:
                    raise ValueError(
                        "Estimate or supply a positive period before fitting a quasi-periodic GP"
                    )
                base = QuasiPeriodicKernel()
                if preset is None:
                    base.kernels[0].period_length = period
                    base.kernels[0].lengthscale = 1.0
                    base.kernels[1].lengthscale = length_scale
                else:
                    # Apply parameter bounds and their specified starting values together.
                    base.kernels[0].register_constraint("raw_period_length", _constraint(preset["period"]))
                    base.kernels[0].register_constraint("raw_lengthscale", _constraint(preset["periodic_lengthscale"]))
                    base.kernels[1].register_constraint("raw_lengthscale", _constraint(preset["rbf_lengthscale"]))
            elif kernel == "rbf":
                base = gpytorch.kernels.RBFKernel()
                base.lengthscale = length_scale
            else:
                raise ValueError("kernel must be 'quasiperiodic', 'rbf', or a GPyTorch kernel")
            if preset is None:
                kernel = gpytorch.kernels.ScaleKernel(base)
                kernel.outputscale = max(float(np.var(flux)), 1e-8)
            else:
                kernel = gpytorch.kernels.ScaleKernel(base, outputscale_constraint=_constraint(preset["outputscale"]))
        else:
            kernel = copy.deepcopy(kernel)
        # Initialize a constant mean or copy the caller-supplied mean module.
        if mean is None:
            if preset is None:
                mean = gpytorch.means.ConstantMean()
                mean.constant = float(np.median(flux))
            else:
                mean = gpytorch.means.ConstantMean(constant_constraint=_constraint(preset["mean"]))
        else:
            mean = copy.deepcopy(mean)
        if noise is not None and (not np.isfinite(noise) or noise <= 0):
            raise ValueError("noise is an additional standard deviation and must be positive")
        # Combine known measurement variance with positive learnable additional noise.
        noise_constraint = (_constraint(preset["noise"]) if preset else gpytorch.constraints.GreaterThan(1e-12))
        if errors is not None:
            # Preserve supplied positive variances instead of applying dtype-dependent floors.
            with gpytorch.settings.min_fixed_noise(float_value=0.0, double_value=0.0):
                likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                    noise=torch.as_tensor(errors**2, dtype=self.torch_dtype, device=device),
                    learn_additional_noise=True,
                    noise_constraint=noise_constraint,
                )
            likelihood.second_noise_covar.noise = (
                .5 if preset else (noise or max(float(np.std(np.diff(flux))) * .1, 1e-5)) ** 2
            )
        else:
            likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
            likelihood.noise = .5 if preset else (noise or max(float(np.std(np.diff(flux))) / np.sqrt(2), 1e-5)) ** 2
        self.model = ExactGPModel(x, y, likelihood, kernel, mean).to(
            device=device, dtype=self.torch_dtype
        )
        # Transfer learned parameters without copying observation-dependent noise buffers.
        if initial_fit is not None:
            previous = dict(initial_fit.model.named_parameters())
            current = dict(self.model.named_parameters())
            if previous.keys() != current.keys():
                raise ValueError("Warm initialization requires compatible GP modules")
            with torch.no_grad():
                for name, parameter in current.items():
                    parameter.copy_(previous[name].to(parameter))
        self.errors = None if errors is None else np.array(errors, dtype=self.numpy_dtype)
        # Optimize exact marginal likelihood while tracking the best parameter state.
        self.model.train()
        likelihood.train()
        opt_class = torch.optim.Adam if optimizer == "adam" else torch.optim.SGD
        opt = opt_class(self.model.parameters(), lr=learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self.model)
        self.losses = []
        best, stopping_best, stale, best_state = np.inf, np.inf, 0, None
        best_iteration, selected_iteration, updates = None, None, 0
        stopped = False
        # Apply warm-up and stop when the non-improvement counter exceeds patience.
        with gpytorch.settings.max_cholesky_size(10000):
            for i in range(iterations):
                opt.zero_grad()
                prior = self.model(x)
                loss = (torch.nn.functional.mse_loss(prior.mean, y)
                        if objective == "mse" else -mll(prior, y))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite GP loss; check input scaling and covariance")
                value = float(loss.detach().cpu())
                self.losses.append(value)
                if value < best:
                    best, best_iteration = value, i
                    best_state = copy.deepcopy(self.model.state_dict())
                if early_stopping and i > min_iterations:
                    if value < stopping_best:
                        stopping_best, stale = value, 0
                    else:
                        stale += 1
                    if stale > patience:
                        stopped = True
                        break
                loss.backward()
                opt.step()
                updates += 1
        # Select parameters explicitly before measuring residuals or building inference caches.
        if save_state == "best":
            self.model.load_state_dict(best_state)
            selected_iteration = best_iteration
        else:
            selected_iteration = updates
        self.model.eval()
        likelihood.eval()
        noise_module = likelihood.second_noise_covar if errors is not None else likelihood.noise_covar
        noise_before = float(noise_module.noise.detach().cpu().item())
        residual_variance = None
        if set_noise_equal_to_var_residuals:
            # Measure the training posterior residuals with the selected, pre-reset model.
            with torch.no_grad():
                prior_mean = self.model.mean_module(x).cpu().numpy()
                kernel_cov = self.model.covar_module(x).to_dense().cpu().numpy()
            diagonal = np.full(len(time), noise_before, dtype=self.numpy_dtype)
            if errors is not None:
                diagonal += np.asarray(errors, dtype=self.numpy_dtype)**2
            factor = numpy_factor(kernel_cov + np.diag(diagonal), self.numerical_diagnostics, "noise-reset covariance")
            posterior = prior_mean + kernel_cov @ cho_solve(factor, self.flux - prior_mean)
            residual_variance = float(np.var(self.flux - posterior))
            # Stay strictly inside the likelihood constraint to avoid infinite raw parameters.
            floor = max(1e-4, float(noise_module.raw_noise_constraint.lower_bound.cpu()))
            floor = np.nextafter(self.numpy_dtype(floor), self.numpy_dtype(np.inf))
            noise_module.noise = max(float(floor), residual_variance)
        # Report likelihood of the final prior-plus-noise model, not a posterior evaluated as a prior.
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(10000):
            self.marginal_log_likelihood = float(mll(self.model.forward(x), y).cpu())
        self.settings = dict(
            kernel=type(kernel).__name__,
            qp_constraints=qp_constraints,
            constraint_settings=preset,
            period=period,
            length_scale=preset["rbf_lengthscale"][2] if preset else length_scale,
            noise=noise,
            iterations=iterations,
            learning_rate=learning_rate,
            optimizer=optimizer,
            patience=patience,
            device=str(device),
            seed=seed,
            objective=objective, early_stopping=early_stopping, min_iterations=min_iterations,
            save_state=save_state, dtype=dtype, best_iteration=best_iteration,
            selected_state_step=selected_iteration, optimizer_steps=updates,
            set_noise_equal_to_var_residuals=set_noise_equal_to_var_residuals,
            noise_before_adjustment=noise_before, residual_variance=residual_variance,
            marginal_log_likelihood=self.marginal_log_likelihood,
            completed_iterations=len(self.losses),
            stopping_reason="patience" if stopped else "iteration_limit",
        )
        # Cache the fitted prior and noise terms without retaining an autograd graph.
        with torch.no_grad():
            self.prior_mean = self.model.mean_module(x).cpu().numpy()
            self.kernel_cov = self.model.covar_module(x).to_dense().cpu().numpy()
            self.noise_variance = float(
                (likelihood.second_noise_covar.noise if errors is not None else likelihood.noise)
                .cpu()
                .reshape(-1)[0]
            )
        self.settings["noise_after_adjustment"] = self.noise_variance
        diagonal = np.full(len(time), self.noise_variance, dtype=self.numpy_dtype)
        if errors is not None:
            diagonal += np.asarray(errors, dtype=self.numpy_dtype)**2
        # Factor the full noisy covariance once for subsequent conditioning.
        self.cov = self.kernel_cov + np.diag(diagonal)
        lower, self.cov = stable_cholesky(self.cov, self.numerical_diagnostics, "fitted observation covariance")
        self.factor = (lower, True)
        self.settings["numerical_diagnostics"] = self.numerical_diagnostics
        self._precision = None
        # Cache the full-data solve and only the most recent masked solve in RAM.
        self._alpha = cho_solve(self.factor, self.flux - self.prior_mean)
        self._subset_cache = None
        self._prediction_cache = None

    def plot(self, axes=None):
        """Plot optimization loss and fitted covariance components without displaying.

        :param axes: Two axes, or four for the built-in QP kernel; None creates a figure.
        :type axes: sequence of matplotlib.axes.Axes or None
        :returns: Figure and axes containing loss and covariance panels.
        :rtype: tuple
        """
        import matplotlib.pyplot as plt

        # Add periodic and evolution panels when the fitted kernel exposes those components.
        base = getattr(self.model.covar_module, "base_kernel", None)
        qp = isinstance(base, QuasiPeriodicKernel)
        count = 4 if qp else 2
        if axes is None:
            _, axes = plt.subplots(1, count, figsize=(5 * count, 4))
        axes = np.asarray(axes).ravel()
        if len(axes) != count:
            raise ValueError(f"Provide {count} axes for this fitted kernel")
        axes[0].plot(self.losses)
        axes[0].set(xlabel="Objective evaluation", ylabel=self.settings["objective"],
                    title=f"Training loss ({self.settings['save_state']} state)")
        matrices = [("Fitted kernel covariance", self.kernel_cov)]
        if qp:
            x = torch.as_tensor(self.time-self.origin, dtype=self.torch_dtype, device=self.device)
            with torch.no_grad():
                matrices.extend((name, kernel(x).to_dense().cpu().numpy()) for name, kernel in
                                zip(("Periodic covariance", "RBF covariance"), base.kernels))
        for ax, (title, matrix) in zip(axes[1:], matrices):
            artist = ax.imshow(matrix, origin="lower", aspect="auto")
            ax.set(title=title, xlabel="Sample", ylabel="Sample")
            ax.figure.colorbar(artist, ax=ax)
        axes[0].figure.tight_layout()
        return axes[0].figure, axes

    @property
    def precision(self):
        """Compute and cache the full inverse observation covariance on first access.

        :returns: Cached full observation precision matrix.
        :rtype: numpy.ndarray
        """
        # Build the expensive precision matrix only when accelerated searches request it.
        if self._precision is None:
            self._precision = cho_solve(self.factor, np.eye(len(self.time), dtype=self.numpy_dtype))
        return self._precision

    def predict(self, time, train_mask=None, include_noise=False, errors=None):
        """Condition the fitted GP on the selected training samples.

        :param time: Timestamps in days, on the input time reference.
        :type time: numpy.ndarray
        :param train_mask: Boolean sample-selection mask; None selects all training samples.
        :type train_mask: array-like or None
        :param include_noise: Include learned additional noise and supplied measurement variance in predictive
            uncertainty.
        :type include_noise: bool
        :param errors: Positive one-sigma measurement errors in flux units; None omits known errors.
        :type errors: numpy.ndarray or None
        :returns: Predictive mean and nonnegative pointwise standard deviation.
        :rtype: tuple
        """
        # Select the retained training samples and evaluate prediction-to-training covariance.
        train = np.arange(len(self.time)) if train_mask is None else np.flatnonzero(train_mask)
        time = np.asarray(time, float)
        x = torch.as_tensor(time - self.origin, dtype=self.torch_dtype, device=self.device)
        tx = torch.as_tensor(
            self.time[train] - self.origin, dtype=self.torch_dtype, device=self.device
        )
        cached = self._prediction_cache
        if cached is not None and np.array_equal(time, cached[0]) and np.array_equal(train, cached[1]):
            prediction, variance = cached[2].copy(), cached[3].copy()
        else:
            # Reuse fitted kernel arrays when predicting on the original grid.
            if np.array_equal(time, self.time):
                prior = self.prior_mean
                cross = self.kernel_cov[:, train]
                diagonal = np.diag(self.kernel_cov).copy()
            else:
                with torch.no_grad():
                    prior = self.model.mean_module(x).cpu().numpy()
                    cross = self.model.covar_module(x, tx).to_dense().cpu().numpy()
                    diagonal = self.model.covar_module(x, diag=True).cpu().numpy()
            if len(train):
                if np.array_equal(train, np.arange(len(self.time))):
                    factor, alpha = self.factor, self._alpha
                elif self._subset_cache is not None and np.array_equal(train, self._subset_cache[0]):
                    _, factor, alpha = self._subset_cache
                else:
                    # Bound memory growth by replacing, rather than accumulating, masked caches.
                    factor = numpy_factor(self.cov[np.ix_(train, train)], self.numerical_diagnostics, "prediction retained covariance")
                    alpha = cho_solve(factor, self.flux[train] - self.prior_mean[train])
                    self._subset_cache = (train.copy(), factor, alpha)
                prediction = prior + cross @ alpha
                variance = diagonal - np.sum(cross * cho_solve(factor, cross.T).T, axis=1)
            else:
                prediction, variance = prior, diagonal
            # Cache latent predictions only; observation noise is added independently below.
            self._prediction_cache = (time.copy(), train.copy(), prediction.copy(), variance.copy())
        # Optionally add observation noise to the latent predictive variance.
        if include_noise:
            variance += self.noise_variance
            if errors is not None:
                variance += np.asarray(errors) ** 2
        return prediction, np.sqrt(np.maximum(variance, 0))
