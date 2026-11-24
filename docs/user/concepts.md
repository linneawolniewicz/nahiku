# Data conventions and interpretation

## Observation and working state

Timestamps must be finite, strictly increasing, and expressed in days. Flux and one-sigma measurement errors use the same linear units. Record the time reference in metadata: Kepler BKJD is BJD − 2454833;
TESS BTJD is BJD − 2457000, both TDB for the supplied examples. An array index is not a time.
The constructor rejects invalid samples and duplicate timestamps. Lightkurve conversion
filters non-finite time/flux and sorts times while recording source row indices.
If any finite positive errors exist, rows with invalid errors are removed; otherwise
measurement errors are treated as unavailable.

Original arrays are retained as read-only data. `standardize()` centers and scales working flux and scales its uncertainties. `prewhiten()` removes sinusoids without automatically
standardizing. `bin(width)` returns a new object and records contributing sample groups.
`reset()` restores originals. For a binned object, these originals are the binned
working values at its construction, not the parent’s unbinned observations. At least
three occupied bins must remain. Data-changing operations clear fits and search results.
Stored prewhitening diagnostics describe the run that produced them; subsequent
standardization does not rescale those diagnostic arrays.

## Variability and preprocessing

`estimate_period()` evaluates a Lomb–Scargle periodogram at the observation timestamps. An inconclusive estimate
returns `None`; do not interpret the strongest periodogram feature automatically as
stellar rotation. `fit_gp(period=...)` accepts a supplied period, and `kernel="rbf"`
models smooth variability without asserting recurrence. Custom GPyTorch kernels and
means are accepted and copied before fitting.

For a heuristic GP initialization when no peak is accepted, request
`estimate_period(fallback="shoulder")`. This smooths the Lomb–Scargle spectrum
with a Gaussian width of two frequency bins and uses its first negative slope.
The returned timescale is capped at the observation duration and must satisfy the
requested period bounds. It is not evidence of a significant periodicity.
`period_result["fallback_used"]` and `period_result["reason"]` identify a fallback;
`peak_accepted` remains false, and `false_alarm_probability` still describes the
original strongest peak. Smoothed power and slopes are retained for inspection.
Constant flux or the absence of a usable spectral shoulder returns `None`. The default,
`fallback=None`, returns only a peak that passes the significance and boundary checks.

Both prewhitening algorithms return the same result type:

```python
result = lc.prewhiten(fmin=20, fmax=80)  # method="balmung" is the default
result = other.prewhiten(fmin=20, fmax=80, method="weighted")
```

Use independent copies or reset observations before comparing methods; calling
prewhitening twice acts on the residuals of the first call.
The default uses Dan Hey's bundled Balmung implementation, checked against the
reference source. Balmung jointly fits frequency,
amplitude, and phase without uncertainty weights,
and uses a fixed threshold from the initial amplitude spectrum. Its frequency
bounds select the initial peak; the subsequent nonlinear fit is unconstrained.
Its internal frequency sampling is fixed at fivefold oversampling. Leave
`samples_per_peak=None` (default) or supply `5`; other values require the weighted
method. Balmung optimizer errors propagate instead of returning partial success.
The one documented numerical patch returns frequency rather than amplitude when
the largest spectrum peak lies at a boundary; see `THIRD_PARTY_NOTICES.md`.
Nāhiku's adapter preserves the continuum, packages diagnostics, and reports stopping
without changing Balmung's component-fitting loop.
Nāhiku's optional weighted method fits an offset plus sine/cosine coefficients at locally
bounded trial frequencies, uses supplied measurement uncertainties, and updates
the spectral threshold after each removal. `fit_mask` requires `method="weighted"`;
Balmung rejects this option. Stored measurement errors remain unchanged for both.
The method is recorded in `result.method` and `lc.history`.
It returns frequencies, amplitudes, phases, reconstruction, residuals, spectra, and a
stopping reason. Its phase origin is the first observation. Reaching the component limit does not establish convergence. Check injected-dip recovery and sensitivity to frequency bounds.
Measurement errors do not include pulsation-fit uncertainty.

## Detection methods

Greedy detection defaults to ordinary posterior residuals (`seed_statistic="residual"`)
and metric-driven exclusion growth (`growth="metric"`, `growth_metric="mll"`).
`direction="both"` compares absolute residuals; select `"negative"` for dips or
`"positive"` for brightenings. `len_deviant` requires that many contiguous samples
above their thresholds and ranks eligible windows by their minimum residual strength.
It applies equally to `seed_statistic="loo"`, which uses leave-one-out standardized
residuals. Masked samples and enabled gap barriers cannot lie inside a seed window.

For ordinary residuals, the significance threshold is
`sigma * sqrt(flux_err**2 + var(abs(residual)))`, with zero measurement error when
none was supplied. `sigma=3` is the default. `update_threshold=False` freezes the
initial residual-scatter term; `True` recomputes it on retained observations after
each accepted event. LOO uses current conditional observation uncertainty and a
fixed `sigma` multiplier instead; `update_threshold=True` is rejected for LOO.
Hyperparameter fitting still uses the observations, even for LOO statistics.

Metric growth conditions on samples outside the candidate interval, adds
`expansion_param` samples at each available edge, and accepts only strictly
improving retained-data losses. It keeps the last accepted interval when an
expansion fails. All growth losses are minimized: `mll` is negative Gaussian prior
marginal log likelihood per retained sample; `nlpd` is joint posterior predictive
negative log density per retained sample; `msll` is pointwise predictive log loss
relative to a Gaussian baseline estimated from retained flux; `rmse` is retained
posterior RMS residual. These are growth criteria, not event significance tests.
Unlike threshold growth, metric growth does not require every added sample to
have the seed's sign or exceed its threshold. Comparisons between different retained subsets are heuristic interval-selection
criteria and require calibration through injection and recovery tests.

Alternatively, `growth="threshold"` extends through same-sign neighbors exceeding
`grow_sigma` (default 2) on the selected statistic's scale, without reconditioning
at every extension. It uses the current iteration's scale throughout growth.
`refit=True` optimizes each metric-growth trial, or each accepted threshold-grown
interval. `warm_initialization=True` carries forward accepted GP parameters;
rejected trial parameters are discarded. Optimizer state starts fresh each time.
Set it to `False` to use the original fitting configuration's initial values.

`max_events=None` runs until no qualifying seed remains, subject to retaining at
least `min_remaining=3` observations. An integer adds a limit on accepted seeds, before adjacent intervals are merged into events. The
stop reason and residual-variance history are recorded in result settings. There
is no gap barrier by default (`gap_factor=None`); for example, `gap_factor=5`
prevents seeds and growth from crossing gaps larger than five median cadences.

Exhaustive detection holds out each configured interval and evaluates the full
conditional observation distribution. The precision-matrix method agrees with direct
conditioning at fixed hyperparameters; it is not a separate approximation or dynamic
programming algorithm. Refitting each interval requires the slower direct method.

| Metric | Stronger anomaly | Meaning |
|---|---|---|
| `pvalue` | Smaller | Chi-square tail of squared conditional Mahalanobis distance |
| `mahalanobis` | Larger | Covariance-normalized residual norm |
| `nlpd` | Larger | Negative joint log predictive density per sample |
| `rmse` | Larger | RMS residual in working flux units |
| `ll` | Smaller | Joint held-out observation log density (not divided by interval length) |
| `mll` | Smaller | Mean pointwise marginal held-out observation log density |
| `msll` | Larger | Mean pointwise negative log density minus a retained-data Gaussian baseline loss |

For exhaustive scoring, `mll` averages the log densities of individual held-out
observation marginals; it includes observation noise once and does not feed a
posterior distribution into a training marginal-likelihood objective. This differs
from greedy growth's `mll`, which evaluates the retained-data **prior** marginal
likelihood. `ll` and `nlpd` retain the full held-out covariance, with
`ll = -number_of_held_out_samples * nlpd`. Joint `ll` depends on interval length;
choose lengths and cutoffs accordingly. MSLL uses the mean and population variance
of the retained flux for its Gaussian baseline (variance floored at machine epsilon).

Exhaustive conditioning defaults to `backend="torch"` on the fitted dtype/device;
`backend="numpy"` selects SciPy on CPU. Both support `acceleration=True` (cached full
precision) and `False` (direct conditioning). Torch tensors and the optional full
precision stay in memory during a search; scalar scores use NumPy/SciPy after the
conditional distribution is transferred to CPU. CUDA speed has not been benchmarked.

With `refit=True`, set `acceleration=False`. `warm_initialization=True` starts **each
candidate independently from the full-data fitted hyperparameters**, so candidate
order does not affect initialization. Set it to `False` for fresh configured values.
Optimizer state is new each time, and fixed measurement errors belong to the retained
observations. The overview prediction uses the full-data fitted hyperparameters conditioned
outside all selected samples. Individual conditional plots retain each candidate's
prediction, including any refit.

Exhaustive sign filtering uses the **mean residual of each window**; it does not
require every sample to have the selected sign. Greedy scores record the minimum
standardized strength of the seed before growth, not a score for the final interval.

Choose a threshold or explicitly request `top_k` non-overlapping candidates. Ranking
is not a significance test. The p-values assume a fitted Gaussian model and do not
account for multiple overlapping tests, hyperparameter uncertainty, or model mismatch.
Calibrate survey thresholds using representative noise-only and injected-signal data.

## Results and evaluation

Candidate and merged event intervals use half-open sample indices `[start, stop)`.
Event exports include the first and last observed timestamps. The exhaustive
overview conditions outside all flagged samples; each candidate score and its
conditional plot use that candidate's excluded interval.
This distinction matters when interpreting residuals and broad merged detections.

Evaluation uses maximum one-to-one overlap matching. A single detection cannot count as
two recovered events. Precision and recall are bounded by one; no-event denominators
return zero. Stellar examples report candidate events without assigning ground-truth accuracy.

Dense exact GPs can absorb anomalies when variability and event timescales overlap.
Neither preprocessing nor a small nominal p-value establishes an astrophysical cause.

## GP initialization and in-memory caching

`fit_gp()` standardizes working flux and measurement errors unless the mean is
within 1e-8 of zero and population standard deviation is within 1e-6 of one.
Original observations are preserved; applied scaling is recorded in history.
Use `standardize=False` to retain the current flux units. Constant flux requires
that opt-out. Supplied noise and mean settings refer to the resulting working units.
A failed fit leaves existing working observations unchanged.

```python
lc.fit_gp(period=2.0, qp_constraints="period-informed")
lc.fit_gp(kernel="rbf", standardize=False)  # explicitly keep current units
```

The optional period-informed preset specifies period-dependent
QP bounds and starting values, plus output variance bounds (0, 5), starting at 1;
constant mean bounds (-1, 1), starting at 0.01; and noise variance above 1e-4,
starting at 0.5. With known errors, this noise is additional to their fixed variance.
It requires the built-in QP kernel and rejects explicit `length_scale`, `noise`,
or `mean` overrides to keep the preset internally consistent. Without the preset,
`length_scale=None` means the usual one-day initial RBF timescale.

| Period P (days) | Period bounds | Periodic lower / initial | RBF lower / initial (days) |
|---|---|---|---|
| P ≤ 0.5 | max(0.001, P−0.1), 2 | P/4 / P | P/4 / 4P |
| 0.5 < P < 1 | 0.4, 5 | 0.1 / P | 1 / 3P |
| 1 ≤ P < 4 | 0.4, 5 | 0.2 / P/2 | 1 / 2P |
| 4 ≤ P < 8 | P−1, P+1 | 0.4 / P/4 | P/3 / 1.5P |
| P ≥ 8 | P−2, P+2 | 0.4 / 2 | P/4 / P |

The periodic lengthscale is dimensionless. Period starts at P in all branches.
Periods whose prescribed initial values violate the parameter bounds are rejected.
The full constraint specification is recorded in fit diagnostics and retained during
subset refits; subsets stay in the full light curve's working units.

Covariance, Cholesky factors, solved residuals, and lazily computed precision
are cached only in memory. Prediction reuses the original-grid covariance,
full-training factors, the latest masked-subset factors, and the latest latent
prediction. The two latter caches each retain only one entry. Learned-noise and
measurement variances are added after retrieving latent predictions. Data changes
invalidate fitted state. Numerical exports do not write these caches to disk.
GPFit caches use NumPy/SciPy on the CPU. Search conditioning also supports Torch
tensors on the fitted device, including CUDA.

## Training controls and residual-based noise adjustment

The default training configuration uses Adam,
1,000 iterations, learning rate 0.01, patience 1, and the final model parameter state.
The default objective is negative marginal log likelihood. Early-stopping checks
start when the zero-based iteration index exceeds `min_iterations`, which defaults
to `iterations // 10`. Any strictly smaller loss improves the tracked value;
stopping occurs when the non-improvement count **exceeds** patience.

```python
lc, final_mll = lc.fit_gp(
    period=2.0,
    qp_constraints="period-informed",
    iterations=1000,
    learning_rate=0.01,
    optimizer="adam",                 # or "sgd"
    objective="mll",                  # or "mse"
    early_stopping=True,
    min_iterations=100,
    patience=1,
    save_state="last",                # or "best"
    set_noise_equal_to_var_residuals=True,
    dtype="float64",                  # or "float32"
    return_mll=True,
)
fig, axes = lc.plot_fit()              # optional; no implicit show()
```

`save_state="best"` restores the lowest evaluated objective state, including
warm-up evaluations; `"last"` keeps the final model parameter state. Neither option saves optimizer
momentum or other Adam/SGD state. Losses are measured
before each update, so the last post-update state at the iteration limit need not
appear in the loss history. `early_stopping=False` performs all requested steps.

`objective="mse"` minimizes the training-mode **prior mean** loss.
With a constant mean it trains that mean, not the covariance or noise parameters;
it is not posterior-prediction MSE. Use MLL to optimize the full covariance model.

The optional noise reset first predicts at training observations using the selected
state, then sets learned variance to the residual population variance with a 1e-4
floor (nudged inside a strict constraint when necessary). Known measurement-error
variances remain fixed and are added separately. It performs no further optimization.
All covariance and prediction caches are constructed after the reset. Diagnostics
record the pre-reset noise, residual variance, and final noise. This adjustment occurs after fitting and does not maximize marginal likelihood.

`return_mll=True` returns `(lc, final_mll)`; otherwise fitting still returns `lc`.
This is GPyTorch's marginal log likelihood normalized per observation, evaluated
on the selected model **after** any noise adjustment. It is also in fit diagnostics.
`plot_fit()` shows loss and kernel covariance, plus periodic/RBF components for the
built-in QP kernel. Precision can be float32 or float64 (default); NumPy/SciPy caches
use the selected dtype and stay in CPU memory even when training uses CUDA.

## Greedy conditioning backends

```python
torch_result = lc.greedy_search()  # default: Torch on the fitted device
numpy_result = lc.greedy_search(backend="numpy")  # NumPy/SciPy on CPU
```

Both backends support the same seeding statistics and apply identical direction,
growth, and observing-gap rules. The Torch backend
uses the fitted model's dtype and device and retains covariance tensors and one
training-subset factorization in memory during the search. Returned results are
NumPy arrays; the backend is recorded in search settings. `refit=True` is supported
by both backends. Switching backends does not select a different detection statistic.
Floating-point differences can affect candidates exactly at a detection threshold.

The Torch backend uses Torch Cholesky solves for conditioning and Gaussian growth
losses. Dense fitted GP caches also remain in CPU memory; results are not written
to disk automatically. Selecting a backend does not change the seed statistic. Numerical tests compare
both implementations against GPyTorch exact posteriors, including held-out boundary
samples, heterogeneous errors, empty training sets, and float32/float64 arithmetic.
CPU behavior is tested; CUDA performance has not been benchmarked.

Float64 is recommended for strongly correlated, low-noise light curves. In a
256-point CPU benchmark with covariance condition number around 2e5, float32
posterior means differed from GPyTorch by up to 0.052 standardized flux units,
while float64 differences were below 6e-9. Matching detection masks does not imply
matching predictions near numerical limits. The small-matrix reference tests use
explicit precision-dependent tolerances; they do not guarantee equivalent rounding
for every covariance matrix.


## Numerical covariance recovery

Nāhiku symmetrizes matrices before Cholesky factorization. If an accelerated
precision block or its resulting conditional covariance cannot be factored without
jitter, it retries that interval directly from the original observation covariance.
Precision blocks are not regularized; failed blocks require direct conditioning.

Covariance factorizations first try no jitter, then bounded positive diagonal
adjustments relative to the largest absolute diagonal entry. The maximum is
`1e-6 * scale`; retry sizes also respect the arithmetic precision. A zero diagonal
uses the largest absolute matrix entry, or unit scale for an entirely zero matrix.
The same policy applies to NumPy and Torch conditioning. Numerical jitter is a
computational perturbation: learned noise parameters and supplied errors are not
changed. Recovered calculations are therefore approximate, and recovery is recorded.

Search results expose `result.settings["numerical_diagnostics"]`: symmetry adjustments,
direct fallbacks with reasons, and successful jitter amounts (absolute and relative)
include the operation or interval context. Fitted covariance adjustments are also
in fit settings and search provenance. Lower-level helpers warn when no diagnostic
collector is supplied. Exporting the result preserves these records.

If the bounded retries fail, `nahiku.numerics.CovarianceError` (a NumPy
`LinAlgError` subclass) reports the stage, dtype, matrix scale, minimum eigenvalue,
and jitter limit. Non-finite matrices fail immediately. No pseudoinverse or generic
solve is used to assign a Gaussian probability to an indefinite covariance. This
recovery policy does not guarantee accurate inference for arbitrary ill-conditioning;
float64 remains preferable for strongly correlated, low-noise observations.


## Inspect exhaustive conditional predictions

`result.plot()` defaults to an exhaustive diagnostic with two panels: the GP
conditioned outside the union of detected samples, followed by the exact
conditional predictions for representative scored windows. Colored points mark
held-out observations and bands show ±2 predictive observation standard deviations.
A representative is the strongest selected candidate within each merged event,
with ties resolved by start and stop indices. It need not span the entire event.

```python
result.plot()                         # One representative per merged event
result.plot(selection="all")          # All selected windows, including overlaps
if result.intervals:
    result.plot(event_index=0)        # One GP across the entire light curve
result.plot(kind="residuals")         # Union-conditioned residual diagnostic
```

The single-event view excludes only that representative window; other detections
remain in the conditioning data. Pointwise uncertainty bands do not display the
correlations used by joint scores. Low p-values express disagreement with the
fitted model, not a calibrated probability that a physical event is present.

`result.conditional_predictions` stores selected candidate bounds, scores, exact
scored means, observation standard deviations and covariances, and full-grid
`full_mean`/`full_std` arrays. Candidate refits are preserved without repeated
optimization. Results remain plottable after the original object changes.

`result.export(prefix)` writes these arrays into the NPZ archive under
`conditional_{index}_{field}` keys; the JSON `conditional_predictions` list maps
indices to candidate bounds and scores. Load NPZ files with `allow_pickle=False`.
Export writes numerical arrays and metadata rather than live models. Until export
is requested, the predictions remain in memory.

Storage grows as O(MN + sum(w_i²)) for M selected windows, N observations and
window lengths w_i. Full-grid predictions add conditioning work. Refitted searches
compute full-grid predictions for eligible candidates before their models leave
scope, including candidates subsequently discarded by top-k selection.


## Prediction errors and numerical exports

`predict(include_noise=True)` uses stored measurement errors when `time=None`.
If timestamps are supplied explicitly, pass `flux_err` to include measurement
errors, even when predicting at `lc.time`. Learned additional noise is included
independently. Returned uncertainties are pointwise standard deviations.

`lc.export(prefix)` saves working and original arrays, injection labels, and
processing provenance. Missing errors are represented by empty arrays. Search
predictions require `result.export(prefix)`; complete prewhitening spectra must
be saved separately from the result object. These exports overwrite files with
the same prefix and do not serialize live models or inference caches.
