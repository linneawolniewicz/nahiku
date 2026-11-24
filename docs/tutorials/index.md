# Feature tutorials

These executable notebooks explain the public API through focused examples.

```{toctree}
:maxdepth: 1

loading
preparation
periods
prewhitening
synthetic-noise
synthetic-gp
injection
gp-fitting
greedy
exhaustive
comparison
evaluation
plotting-export
```

Each tutorial records the package version and runs with the scientific dependencies installed.

## Feature coverage

| Public capability | Tutorial |
|---|---|
| Construction, `from_lightkurve`, archive loading | [Loading](loading.ipynb), main real-star examples |
| Original/working arrays, `standardize`, `bin`, `reset`, `copy` | [Preparation](preparation.ipynb) |
| `estimate_period` and significance diagnostics | [Periods](periods.ipynb) |
| `prewhiten`, removed components and spectra | [Prewhitening](prewhitening.ipynb) |
| `from_synthetic_noise` | [Noise generator](synthetic-noise.ipynb) |
| `from_synthetic_gp` | [GP generator](synthetic-gp.ipynb) |
| `inject_anomaly`: Gaussian, saw, exocomet | [Injection](injection.ipynb) |
| `fit_gp`, custom kernels/means, `predict`, fit diagnostics | [GP fitting](gp-fitting.ipynb) |
| `greedy_search` | [Greedy](greedy.ipynb) |
| `exhaustive_search`, scoring, refit, acceleration | [Exhaustive](exhaustive.ipynb) |
| Independent results, threshold interpretation | [Comparison](comparison.ipynb) |
| `evaluate`, one-to-one event matching | [Evaluation](evaluation.ipynb) |
| Object/result `plot` and `export` | [Plotting and exports](plotting-export.ipynb) |
