"""Compare conditioning to GPyTorch exact posteriors on small synthetic problems."""

from copy import deepcopy
from contextlib import ExitStack
import numpy as np
import pytest
import torch
import gpytorch
from nahiku import Nahiku
from nahiku.gp_helpers import condition, condition_precision
from nahiku.torch_backend import TorchConditioner


class MatrixKernel(gpytorch.kernels.Kernel):
    """Fixed finite covariance indexed by integer sample locations."""
    def __init__(self, matrix):
        super().__init__()
        self.register_buffer('matrix', matrix)

    def forward(self, x1, x2, diag=False, **params):
        i,j=x1[...,0].long(),x2[...,0].long()
        block=self.matrix[i[:,None],j[None,:]]
        return block.diagonal() if diag else block


class VectorMean(gpytorch.means.Mean):
    def __init__(self, values):
        super().__init__()
        self.register_buffer('values',values)

    def forward(self,x):
        return self.values[x[...,0].long()]


class ReferenceGP(gpytorch.models.ExactGP):
    def __init__(self,x,y,likelihood,kernel,mean):
        super().__init__(x,y,likelihood)
        self.kernel=kernel
        self.mean=mean

    def forward(self,x):
        return gpytorch.distributions.MultivariateNormal(self.mean(x),self.kernel(x))


def reference(mean,kernel,flux,noise,train,test,dtype):
    """Use GPyTorch's own exact prediction machinery, with no Nahiku conditioning."""
    tensor=lambda a:torch.as_tensor(a,dtype=dtype)
    with gpytorch.settings.min_fixed_noise(float_value=0,double_value=0):
        likelihood=gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=tensor(noise[train]))
    model=ReferenceGP(tensor(train),tensor(flux[train]),likelihood,
                      MatrixKernel(tensor(kernel)),VectorMean(tensor(mean))).to(dtype=dtype)
    model.eval();likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.fast_computations(covar_root_decomposition=False,log_prob=False,solves=False):
        latent=model(tensor(test)) if len(train) else model.forward(tensor(test))
        result=likelihood(latent,noise=tensor(noise[test]))
        return result.mean.numpy(),result.covariance_matrix.numpy()


@pytest.mark.parametrize('dtype,tol',[(torch.float64,1e-9),(torch.float32,2e-5)])
@pytest.mark.parametrize('test',[[0],[8],[2,3,4],[0,4,8]])
def test_random_spd_conditioning_against_gpytorch(dtype,tol,test):
    rng=np.random.default_rng(43);a=rng.normal(size=(9,9));kernel=a@a.T/9+np.eye(9)*.3
    mean=rng.normal(size=9);flux=rng.normal(size=9);noise=np.linspace(.05,.2,9)
    dt=np.float64 if dtype==torch.float64 else np.float32
    kernel,mean,flux,noise=[v.astype(dt) for v in [kernel,mean,flux,noise]]
    cov=kernel+np.diag(noise);test=np.array(test);train=np.setdiff1d(np.arange(9),test)
    expected=reference(mean,kernel,flux,noise,train,test,dtype)
    for actual in [condition(cov,mean,flux,train,test),condition_precision(np.linalg.inv(cov),mean,flux,test)]:
        np.testing.assert_allclose(actual[0],expected[0],atol=tol,rtol=tol)
        np.testing.assert_allclose(actual[1],expected[1],atol=tol,rtol=tol)


@pytest.mark.parametrize('dtype,tol',[('float64',1e-8),('float32',5e-5)])
@pytest.mark.parametrize('retained',[[0,1,2,3,4,5,6,7],[0,2,3,6,7],[2],[]])
def test_predictions_and_loo_against_gpytorch(dtype,tol,retained):
    lc=Nahiku(np.array([0,.1,.4,.9,1.,2.,2.3,3.]),np.array([.1,.3,-.1,.8,.3,-.4,.1,.5]),
              np.linspace(.05,.2,8)).fit_gp(kernel='rbf',iterations=2,dtype=dtype)
    fit=lc._fit;backend=TorchConditioner(fit);train=np.array(retained,dtype=int)
    mask=np.zeros(8,bool);mask[train]=True
    noise=fit.noise_variance+fit.errors**2
    expected=reference(fit.prior_mean,fit.kernel_cov,fit.flux,noise,train,np.arange(8),fit.torch_dtype)
    for predictor in [fit,backend]:
        mean,std=predictor.predict(fit.time,mask,include_noise=True,errors=fit.errors)
        np.testing.assert_allclose(mean,expected[0],atol=tol,rtol=tol)
        np.testing.assert_allclose(std**2,np.diag(expected[1]),atol=tol,rtol=tol)
    if len(train):
        z=backend.loo(mask)
        for position,held in enumerate(train):
            m,c=reference(fit.prior_mean,fit.kernel_cov,fit.flux,noise,train[train!=held],np.array([held]),fit.torch_dtype)
            np.testing.assert_allclose(z[position],(fit.flux[held]-m[0])/np.sqrt(c[0,0]),atol=tol,rtol=tol)


@pytest.mark.parametrize('statistic,growth', [('residual','metric'),('residual','threshold'),('loo','metric'),('loo','threshold')])
@pytest.mark.parametrize('refit',[False,True])
@pytest.mark.parametrize('dtype,tol',[('float64',1e-7),('float32',2e-3)])
def test_greedy_backends_agree(refit,dtype,tol,statistic,growth):
    lc=Nahiku.from_synthetic_noise(n_points=60,red_noise=0,white_noise=.03)
    lc.inject_anomaly(center=5,width=.15,depth=1.5)
    lc.fit_gp(period=2,iterations=5,dtype=dtype,noise=.02,length_scale=5)
    options=dict(sigma=3,grow_sigma=1.5,max_events=3,refit=refit,seed_statistic=statistic,growth=growth)
    a=lc.greedy_search(**options,backend='numpy');b=lc.greedy_search(**options,backend='torch')
    assert a.mask.any()
    assert a.intervals==b.intervals
    np.testing.assert_allclose(a.mean,b.mean,atol=tol,rtol=tol)
    np.testing.assert_allclose(a.std,b.std,atol=tol,rtol=tol)
    np.testing.assert_allclose([c['score'] for c in a.candidates],[c['score'] for c in b.candidates],atol=tol,rtol=tol)
    assert b.settings['backend']=='torch'
    with pytest.raises(ValueError,match='backend'):lc.greedy_search(backend='invalid')
