"""Torch conditioning and leave-one-out statistics for a fixed fitted GP."""

import numpy as np
import torch
from .numerics import stable_cholesky, CovarianceError, record_recovery


class TorchConditioner:
    """Retain covariance tensors and one training-subset factorization in memory.

    :param fit: Fitted GP supplying parameters, observations, dtype, and device.
    :type fit: GPFit
    :param diagnostics: Search recovery records; None records on the fitted GP.
    :type diagnostics: list or None
    """

    def __init__(self, fit, diagnostics=None):
        """Copy fitted numerical state onto the configured Torch device.

        :param fit: Fitted GP with fixed hyperparameters.
        :type fit: GPFit
        :param diagnostics: Search recovery records; None records on the fitted GP.
        :type diagnostics: list or None
        :returns: None.
        :rtype: None
        """
        self.fit = fit
        self.diagnostics = fit.numerical_diagnostics if diagnostics is None else diagnostics
        options = dict(dtype=fit.torch_dtype, device=fit.device)
        self.cov = torch.as_tensor(fit.cov, **options)
        self.kernel_cov = torch.as_tensor(fit.kernel_cov, **options)
        self.mean = torch.as_tensor(fit.prior_mean, **options)
        self.flux = torch.as_tensor(fit.flux, **options)
        self._cache = None

    def _training(self, train_mask):
        """Reuse the factorization and residual solve for the selected training grid.

        :param train_mask: Boolean mask of retained observations; None retains all.
        :type train_mask: array-like or None
        :returns: Integer indices, Cholesky factor, solved residuals, and optional precision.
        :rtype: dict
        """
        indices = (
            np.arange(len(self.fit.time)) if train_mask is None else np.flatnonzero(train_mask)
        )
        if self._cache is not None and np.array_equal(indices, self._cache["indices"]):
            return self._cache
        index = torch.as_tensor(indices, dtype=torch.long, device=self.fit.device)
        block = self.cov[index[:, None], index[None, :]]
        factor, _ = stable_cholesky(block, self.diagnostics, "Torch prediction retained covariance")
        alpha = torch.cholesky_solve((self.flux[index] - self.mean[index])[:, None], factor)[:, 0]
        # Replace the previous subset to bound memory as the event mask changes.
        self._cache = dict(indices=indices, index=index, factor=factor, alpha=alpha, precision=None)
        return self._cache

    @torch.no_grad()
    def loo(self, train_mask=None):
        """Compute leave-one-out standardized residuals for retained observations.

        :param train_mask: Boolean selection on the fitted observation grid, or None for all.
        :type train_mask: array-like or None
        :returns: Standardized residuals in retained-sample order.
        :rtype: numpy.ndarray
        """
        state = self._training(train_mask)
        if state["precision"] is None:
            state["precision"] = torch.cholesky_inverse(state["factor"])
        z = state["alpha"] / torch.sqrt(torch.diagonal(state["precision"]))
        return z.cpu().numpy()

    @torch.no_grad()
    def predict(self, time, train_mask=None, include_noise=False, errors=None):
        """Condition the fitted kernel on retained observations using Torch solves.

        :param time: Prediction timestamps in days.
        :type time: array-like
        :param train_mask: Boolean selection on the fitted grid; None retains all observations.
        :type train_mask: array-like or None
        :param include_noise: Include learned additional noise and supplied measurement variances.
        :type include_noise: bool
        :param errors: Prediction measurement standard deviations, or None.
        :type errors: array-like or None
        :returns: Predictive mean and pointwise standard deviation as CPU arrays.
        :rtype: tuple
        """
        time = np.asarray(time)
        state = self._training(train_mask)
        index = state["index"]
        # Reuse the observed-grid kernel; evaluate new locations with the fitted modules.
        if np.array_equal(time, self.fit.time):
            prior = self.mean
            cross = self.kernel_cov[:, index]
            diagonal = torch.diagonal(self.kernel_cov)
        else:
            options = dict(dtype=self.fit.torch_dtype, device=self.fit.device)
            x = torch.as_tensor(time - self.fit.origin, **options)
            tx = torch.as_tensor(self.fit.time[state["indices"]] - self.fit.origin, **options)
            prior = self.fit.model.mean_module(x)
            cross = self.fit.model.covar_module(x, tx).to_dense()
            diagonal = self.fit.model.covar_module(x, diag=True)
        mean = prior + cross @ state["alpha"]
        variance = diagonal - (cross * torch.cholesky_solve(cross.T, state["factor"]).T).sum(dim=1)
        if include_noise:
            variance = variance + self.fit.noise_variance
            if errors is not None:
                variance = (
                    variance
                    + torch.tensor(
                        np.asarray(errors), dtype=variance.dtype, device=variance.device
                    ).square()
                )
        return mean.cpu().numpy().copy(), variance.clamp_min(0).sqrt().cpu().numpy()


@torch.no_grad()
def condition_torch(covariance, mean, flux, train, test, precision=None, diagnostics=None):
    """Condition a full-grid observation distribution using Torch Cholesky solves.

    :param covariance: Full observation covariance tensor on the chosen device.
    :type covariance: torch.Tensor
    :param mean: Full-grid prior mean tensor.
    :type mean: torch.Tensor
    :param flux: Full-grid observed flux tensor.
    :type flux: torch.Tensor
    :param train: Retained integer indices; must complement test when precision is supplied.
    :type train: array-like
    :param test: Held-out integer indices.
    :type test: array-like
    :param precision: Cached full observation precision tensor, or None for direct conditioning.
    :type precision: torch.Tensor or None
    :param diagnostics: Mutable numerical recovery records; None emits warnings.
    :type diagnostics: list or None
    :returns: Conditional mean and full observation covariance as NumPy arrays.
    :rtype: tuple
    """
    train = torch.as_tensor(train, dtype=torch.long, device=mean.device)
    test = torch.as_tensor(test, dtype=torch.long, device=mean.device)
    context = f"Torch interval {test.tolist()}"
    if precision is not None:
        try:
            # Do not jitter a precision block; retry from original covariance if it fails.
            factor, _ = stable_cholesky(
                precision[test[:, None], test[None, :]],
                diagnostics,
                context + " precision block",
                allow_jitter=False,
            )
            conditional = torch.cholesky_inverse(factor)
            cross = (
                precision[test[:, None], train[None, :]]
                + precision[train[:, None], test[None, :]].T
            ) / 2
            rhs = cross @ (flux[train] - mean[train])
            prediction = mean[test] - torch.cholesky_solve(rhs[:, None], factor)[:, 0]
            _, conditional = stable_cholesky(
                conditional, diagnostics, context + " precision conditional", allow_jitter=False
            )
            if not bool(torch.isfinite(prediction).all()):
                raise CovarianceError(context + ": non-finite conditional mean")
            return prediction.cpu().numpy(), conditional.cpu().numpy()
        except CovarianceError as error:
            record_recovery(
                diagnostics,
                action="direct_fallback",
                context=context,
                backend="torch",
                reason=str(error),
            )
    # Recover directly from the original covariance, with bounded covariance-space jitter.
    asymmetry = float((covariance - covariance.T).abs().max())
    if asymmetry:
        record_recovery(
            diagnostics,
            action="symmetrize",
            context=context,
            backend="torch",
            max_asymmetry=asymmetry,
        )
    covariance = (covariance + covariance.T) / 2
    if len(train):
        factor, _ = stable_cholesky(
            covariance[train[:, None], train[None, :]],
            diagnostics,
            context + " retained covariance",
        )
        cross = covariance[test[:, None], train[None, :]]
        prediction = (
            mean[test]
            + cross @ torch.cholesky_solve((flux[train] - mean[train])[:, None], factor)[:, 0]
        )
        conditional = covariance[test[:, None], test[None, :]] - cross @ torch.cholesky_solve(
            cross.T, factor
        )
    else:
        prediction = mean[test]
        conditional = covariance[test[:, None], test[None, :]]
    _, conditional = stable_cholesky(conditional, diagnostics, context + " conditional covariance")
    if not bool(torch.isfinite(prediction).all()):
        raise CovarianceError(context + ": non-finite conditional mean")
    return prediction.cpu().numpy(), conditional.cpu().numpy()
