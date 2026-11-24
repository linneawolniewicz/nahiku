"""Period-informed constraints, standardized fitting, and in-memory cache reuse."""

import numpy as np
import pytest
from nahiku import Nahiku
from nahiku.gp_helpers import GPFit, condition


@pytest.mark.parametrize('period,bounds,periodic,rbf', [
    (.25,(.15,2),(.0625,.25),(.0625,1)),
    (.5,(.4,2),(.125,.5),(.125,2)),
    (.75,(.4,5),(.1,.75),(1,2.25)),
    (1,(.4,5),(.2,.5),(1,2)),
    (2,(.4,5),(.2,1),(1,4)),
    (4,(3,5),(.4,1),(4/3,6)),
    (8,(6,10),(.4,2),(2,8)),
])
def test_exact_original_initialization(period,bounds,periodic,rbf):
    lc = Nahiku.from_synthetic_noise(n_points=24)
    lc.fit_gp(period=period,qp_constraints='period-informed',iterations=1,save_state='best')
    fit=lc._fit; kernel=fit.model.covar_module; pk,rk=kernel.base_kernel.kernels
    assert pk.period_length.item() == pytest.approx(period,rel=1e-6)
    c=pk.raw_period_length_constraint
    assert c.lower_bound.item() == pytest.approx(bounds[0])
    assert c.upper_bound.item() == pytest.approx(bounds[1])
    assert pk.raw_lengthscale_constraint.lower_bound.item() == pytest.approx(periodic[0])
    assert pk.lengthscale.item() == pytest.approx(periodic[1],rel=1e-6)
    assert rk.raw_lengthscale_constraint.lower_bound.item() == pytest.approx(rbf[0])
    assert rk.lengthscale.item() == pytest.approx(rbf[1],rel=1e-6)
    assert kernel.outputscale.item() == pytest.approx(1,rel=1e-6)
    assert kernel.raw_outputscale_constraint.upper_bound.item() == 5
    assert fit.model.mean_module.constant.item() == pytest.approx(.01,abs=1e-7)
    assert fit.noise_variance == pytest.approx(.5,rel=1e-6)
    assert lc._refit_subset(np.arange(20)).settings['constraint_settings']==fit.settings['constraint_settings']


def test_standardization_default_opt_out_and_atomic_failure():
    x=np.linspace(0,10,40);y=100+3*np.sin(x);e=np.ones(40)*.5
    lc=Nahiku(x,y,e)
    with pytest.raises(ValueError):
        lc.fit_gp(period=2,qp_constraints='period-informed',noise=.2)
    np.testing.assert_array_equal(lc.flux,y)
    lc.fit_gp(kernel='rbf',iterations=2)
    assert abs(lc.flux.mean())<1e-12
    assert lc.flux.std()==pytest.approx(1)
    np.testing.assert_allclose(lc.flux_err,e/y.std())
    np.testing.assert_array_equal(lc.original_flux,y)
    n=len(lc.history)
    lc.fit_gp(kernel='rbf',iterations=2)
    assert len(lc.history)==n
    other=Nahiku(x,y,e).fit_gp(kernel='rbf',iterations=2,standardize=False)
    np.testing.assert_array_equal(other.flux,y)
    np.testing.assert_array_equal(other.flux_err,e)
    assert other.history==[]


def test_estimated_period_survives_auto_standardization():
    x=np.linspace(0,20,200);lc=Nahiku(x,100+3*np.sin(2*np.pi*x/2))
    period=lc.estimate_period(minimum_period=1,maximum_period=3)
    lc.fit_gp(qp_constraints='period-informed',iterations=1,save_state='best')
    assert lc._fit.settings['period']==period==lc.period


def test_prediction_reuses_factors_and_matches_reference(monkeypatch):
    lc=Nahiku.from_synthetic_noise(n_points=30).fit_gp(period=2,iterations=2)
    fit=lc._fit
    import nahiku.gp_helpers as helpers
    real=helpers.numpy_factor;calls=[]
    def tracked(*args,**kwargs):
        calls.append(1)
        return real(*args,**kwargs)
    monkeypatch.setattr(helpers,'numpy_factor',tracked)
    for _ in range(2):fit.predict(fit.time)
    assert calls==[]
    mask=np.ones(30,bool);mask[10:15]=False
    mean,std=fit.predict(fit.time,mask,include_noise=True,errors=fit.errors)
    fit.predict(fit.time,mask,include_noise=True,errors=fit.errors)
    assert len(calls)==1
    held=np.arange(10,15)
    expected,cov=condition(fit.cov,fit.prior_mean,fit.flux,np.flatnonzero(mask),held)
    np.testing.assert_allclose(mean[held],expected,atol=1e-8)
    np.testing.assert_allclose(std[held]**2,np.diag(cov),atol=1e-8)
    assert fit.precision is fit.precision
    lc.standardize()
    assert lc._fit is None


def test_constant_flux_requires_opt_out():
    lc=Nahiku(np.arange(6.),np.ones(6))
    with pytest.raises(ValueError,match='standardize=False'):lc.fit_gp(kernel='rbf',iterations=1)
    lc.fit_gp(kernel='rbf',iterations=1,standardize=False)
