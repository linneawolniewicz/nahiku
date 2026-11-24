"""Optimization policy, residual-noise adjustment, and precision regressions."""

import numpy as np
import pytest
from nahiku import Nahiku


def curve():
    return Nahiku.from_synthetic_noise(n_points=45, red_noise=0, white_noise=.02)


def test_original_stopping_counter_and_warmup():
    x=np.arange(10.);y=np.tile([-1.,1.],5)
    n=Nahiku(x,y).fit_gp(kernel='rbf',objective='mse',iterations=20,
                         min_iterations=2,patience=1)
    d=n.fit_diagnostics
    assert d['optimizer_steps']==5
    assert d['completed_iterations']==6
    assert d['stopping_reason']=='patience'
    assert d['learning_rate']==.01 and d['save_state']=='last'
    n.fit_gp(kernel='rbf',objective='mse',iterations=9,early_stopping=False)
    assert n.fit_diagnostics['optimizer_steps']==9
    assert n.fit_diagnostics['stopping_reason']=='iteration_limit'


def test_explicit_state_selection():
    best=curve().fit_gp(period=2,iterations=1,save_state='best')
    last=curve().fit_gp(period=2,iterations=1,save_state='last')
    assert best.fit_diagnostics['selected_state_step']==0
    assert last.fit_diagnostics['selected_state_step']==1
    a=best._fit.model.covar_module.outputscale.item()
    b=last._fit.model.covar_module.outputscale.item()
    assert a != pytest.approx(b)


@pytest.mark.parametrize('dtype,tolerance',[('float32',2e-4),('float64',1e-9)])
def test_noise_reset_rebuilds_caches_and_final_likelihood(dtype,tolerance):
    options=dict(period=2,iterations=8,dtype=dtype,qp_constraints='period-informed')
    before=curve().fit_gp(**options)
    mean,_=before.predict()
    variance=np.var(before.flux-mean)
    n,mll=curve().fit_gp(**options,set_noise_equal_to_var_residuals=True,return_mll=True)
    fit=n._fit
    assert fit.noise_variance==pytest.approx(max(1e-4,variance),rel=tolerance,abs=tolerance)
    assert fit.noise_variance < before._fit.noise_variance
    assert fit.settings['residual_variance']==pytest.approx(variance,abs=tolerance)
    np.testing.assert_allclose(np.diag(fit.cov-fit.kernel_cov),
                               fit.noise_variance+n.flux_err**2,atol=tolerance)
    r=fit.flux-fit.prior_mean
    expected=-.5*(r@np.linalg.solve(fit.cov,r)+np.linalg.slogdet(fit.cov)[1]+
                   len(r)*np.log(2*np.pi))/len(r)
    assert mll==pytest.approx(expected,abs=tolerance)
    assert fit.cov.dtype==np.dtype(dtype)
    assert fit.precision.dtype==np.dtype(dtype)
    predicted,_=n.predict()
    np.testing.assert_allclose(predicted,fit.prior_mean+fit.kernel_cov@np.linalg.solve(fit.cov,r),
                               atol=tolerance)
    assert n._refit_subset(np.arange(35)).settings['set_noise_equal_to_var_residuals']


def test_mse_updates_mean_not_covariance():
    from nahiku.gp_helpers import GPFit
    x=np.arange(10.);y=np.r_[np.zeros(9),10.]
    initial=GPFit(x,y,None,kernel='rbf',objective='mse',iterations=1,save_state='best')
    final=GPFit(x,y,None,kernel='rbf',objective='mse',iterations=10,early_stopping=False)
    assert final.model.mean_module.constant.item()>initial.model.mean_module.constant.item()
    np.testing.assert_allclose(final.kernel_cov,initial.kernel_cov)
    assert final.noise_variance==initial.noise_variance


def test_plotting_is_optional_and_default_return_is_object():
    import matplotlib.pyplot as plt
    n=curve()
    assert n.fit_gp(period=2,iterations=2) is n
    figure,axes=n.plot_fit()
    assert len(axes)==4
    plt.close(figure)


@pytest.mark.parametrize('option',[dict(objective='bad'),dict(save_state='bad'),dict(dtype='fp16'),
                                    dict(min_iterations=-1),dict(patience=-1)])
def test_invalid_training_option_does_not_transform_data(option):
    n=curve();original=n.flux.copy()
    with pytest.raises(ValueError):n.fit_gp(period=2,iterations=2,**option)
    np.testing.assert_array_equal(n.flux,original)


def test_noise_reset_without_measurement_errors():
    source=curve()
    lc=Nahiku(source.time,source.flux).fit_gp(period=2,iterations=4,
        set_noise_equal_to_var_residuals=True)
    fit=lc._fit
    np.testing.assert_allclose(np.diag(fit.cov-fit.kernel_cov),fit.noise_variance)
    assert np.isfinite(lc.fit_diagnostics['marginal_log_likelihood'])


def test_float32_retains_small_known_errors():
    lc=curve()
    lc=Nahiku(lc.time,lc.flux,np.full(len(lc.time),1e-7))
    lc.fit_gp(period=2,iterations=1,dtype='float32')
    stored=lc._fit.model.likelihood.noise_covar.noise.detach().cpu().numpy()
    np.testing.assert_allclose(stored,lc.flux_err**2,rtol=1e-6,atol=0)
