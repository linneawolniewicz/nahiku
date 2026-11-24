# Developer guide

## Environment and checks

Install `python -m pip install -e ".[dev]"` in Python 3.12. Run `python -m pytest`
from the repository root. Tests import the installed package with Astropy and GPyTorch;
offline tests replace archive calls, and selected tests inject failures or count
internal calls to verify control flow. Numerical reference tests use the scientific
libraries directly. The suite uses neither source-path modification nor global
replacements for scientific libraries.

```bash
python -m pytest
python -m build
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Routine CI executes the synthetic example and all feature tutorials before building docs.
A manually triggered integration job runs the real-star notebooks with Lightkurve/MAST.
The canonical examples are symlinked into `docs/examples`, so there is one notebook source.
Read the Docs renders the saved notebook outputs; it does not download mission data during builds.

## Coverage

Run the same branch-aware coverage check used in CI:

```bash
python -m pytest --cov=nahiku --cov-branch --cov-report=term-missing --cov-report=html --cov-report=xml
```

Open `htmlcov/index.html` to inspect untested statements and branch outcomes. CI
requires at least 90% combined statement/branch coverage and uploads the HTML and
XML reports with its artifacts. Coverage includes the shipped Balmung implementation;
only `exhaustive_helpers.py` is omitted because it is excluded from both distributions.
The remaining retained source can be audited separately with a coverage configuration
that does not omit it. Coverage measures execution, not scientific correctness: retain
independent numerical references, recovery checks, and injection tests alongside it.
CUDA execution and live archive availability require separate tests.

## Numerical stress regressions

Run `python -m pytest tests/test_gp_stress.py tests/test_numerical_recovery.py` for
GP-generated stress cases and recovery-mechanism checks. The stress suite evaluates
real GPyTorch RBF and quasi-periodic kernels on dense, uneven observations, using
float32/float64, NumPy/Torch, and direct/cached-precision conditioning on CPU.
Left/right boundaries, one-point intervals, and long interior intervals are covered.

The accuracy tests use covariance condition numbers around 1e3–2e4 and compare
means, full covariances, and NLPD scores against independent float64 GPyTorch exact
posteriors. Float32 error budgets are below one percent of conditional uncertainty
and variance scales. Float64 additionally must resolve low-noise matrices with
condition numbers above 1e9, without jitter.

Separate float32 cases put noise below representable resolution. These require
bounded, recorded recovery, finite predictions, and positive-definite returned
covariances; they cannot establish agreement with the original model at noise levels that
float32 cannot represent. Near-coincident, distinct timestamps exercise explicit failure when
jitter is disallowed and recorded recovery when it is permitted. Mechanism tests
also cover direct fallback and rejection after the jitter budget is exhausted.

End-to-end fitted light curves contain a known dip: exhaustive direct/accelerated
searches must select its interval, and greedy LOO/threshold searches must recover it
without flagging distant baseline samples. Neither fitted covariance nor learned
noise may change during these searches. These deterministic synthetic regressions do not reproduce historical mission-data
failures or measure CUDA performance.

## Architecture

`Nahiku` owns original/working arrays, history, fit state, and independent search snapshots.
`gp_helpers` uses GPyTorch for hyperparameter training. Dense conditioning supports Torch
(default for searches) and NumPy/SciPy; `numerics` implements bounded covariance recovery.
`Nahiku.estimate_period` records period diagnostics; `preprocessing` implements
sinusoid removal and its frequency diagnostics; detection modules implement the two searches;
`results` handles plotting and portable exports. Internals may evolve without compatibility promises.

All candidate indices are half-open. Preserve full conditional covariance for joint scoring,
include measurement noise, and validate fast calculations against direct conditioning.
Record transformations and invalidate derived state when data change. Scientific changes need numerical
or injection/recovery tests rather than only mocked branch coverage.

## Contributing and support

Open a focused issue or pull request with a reproducible example. Explain scientific assumptions,
include a regression test, and update the corresponding tutorial/API docstring. Use respectful,
constructive discussion. The maintainer reviews contributions; there is no guaranteed response time.
See the repository's `CONTRIBUTING.md` for the complete local workflow.

## Release preparation

Check installation from wheel/sdist outside the source tree, numerical tests, notebook outputs,
strict docs build, fixture checksums, and citation metadata. Review `JOSS_CHECKLIST.md` with the
authors. Verify the contents of the wheel and source distribution. Update version,
changelog, and archive metadata together. Release tagging, uploads, documentation-service activation,
and DOI registration require the maintainer's separate approval. These actions are outside local release preparation.
