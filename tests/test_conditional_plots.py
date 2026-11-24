"""Check stored distributions against the actual exhaustive scoring inputs."""
import importlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from test_exhaustive_options import fitted

@pytest.mark.parametrize('backend', ['numpy', 'torch'])
@pytest.mark.parametrize('refit,acceleration', [(False, True), (False, False), (True, False)])
def test_saved_distributions(backend, refit, acceleration, monkeypatch, tmp_path):
    lc = fitted()
    # Retain test-only references to independently check full-grid conditional moments.
    refits = {}
    original_refit = lc._refit_subset
    def record_refit(indices, initial_fit=None):
        fit = original_refit(indices, initial_fit=initial_fit)
        refits[tuple(indices)] = fit
        return fit
    monkeypatch.setattr(lc, '_refit_subset', record_refit)
    module = importlib.import_module('nahiku.exhaustive_search')
    original = module.interval_score
    scored = []
    def capture(observed, mean, cov, *args, **kwargs):
        scored.append((mean.copy(), cov.copy()))
        return original(observed, mean, cov, *args, **kwargs)
    monkeypatch.setattr(module, 'interval_score', capture)
    result = lc.exhaustive_search(min_points=2, max_points=3, step=8,
        threshold=None, top_k=2, direction='both', backend=backend,
        refit=refit, acceleration=acceleration)
    for record in result.conditional_predictions:
        idx = next(i for i,c in enumerate(result.candidates)
                   if (c['start'],c['stop']) == (record['start'],record['stop']))
        np.testing.assert_array_equal(record['mean'], scored[idx][0])
        np.testing.assert_array_equal(record['covariance'], scored[idx][1])
        np.testing.assert_allclose(record['std']**2, np.diag(scored[idx][1]))
        a,b=record['start'],record['stop']
        np.testing.assert_array_equal(record['full_mean'][a:b], record['mean'])
        assert record['full_mean'].shape == result.time.shape
        # Use a direct dense solve, independently of the plotting prediction path.
        train = np.r_[0:a,b:len(result.time)]
        current = refits[tuple(train)] if refit else lc._fit
        covariance, prior = lc._full_prior(current)
        kernel = covariance.copy()
        kernel[np.diag_indices_from(kernel)] -= current.noise_variance + lc.flux_err**2
        cross = kernel[:, train]
        retained_cov = covariance[np.ix_(train,train)]
        expected_mean = prior + cross @ np.linalg.solve(retained_cov, result.flux[train]-prior[train])
        expected_var = np.diag(covariance) - np.sum(cross * np.linalg.solve(retained_cov,cross.T).T,axis=1)
        np.testing.assert_allclose(record['full_mean'], expected_mean, atol=1e-8)
        np.testing.assert_allclose(record['full_std']**2, expected_var, atol=1e-8)
        assert original(result.flux[a:b], record['mean'],record['covariance'],'pvalue') == pytest.approx(record['score'])
    lc.reset()
    for options in ({}, {'selection':'all'}, {'event_index':0}, {'kind':'residuals'}):
        fig, axes = result.plot(**options)
        assert len(axes) == (1 if 'event_index' in options else 2)
        plt.close(fig)
    paths=result.export(tmp_path/'result')
    arrays=np.load(paths['arrays'], allow_pickle=False)
    np.testing.assert_array_equal(arrays['conditional_0_full_mean'],result.conditional_predictions[0]['full_mean'])
    assert json.loads(paths['metadata'].read_text())['conditional_predictions'][0]['index']==0


def test_empty_and_invalid_views():
    result=fitted().exhaustive_search(min_points=2,max_points=2,threshold=0)
    assert not result.conditional_predictions
    fig,_=result.plot();plt.close(fig)
    for options in ({'event_index':0},{'kind':'invalid'},{'selection':'invalid'}):
        with pytest.raises(ValueError): result.plot(**options)
