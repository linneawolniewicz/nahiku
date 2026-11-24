# Nāhiku

Nāhiku is the ʻŌlelo Hawaiʻi name for the Big Dipper. This Python package detects
anomalous intervals in stellar light curves using Gaussian-process models of background
variability. It provides prewhitening, synthetic injection and recovery tests, and greedy or
exhaustive searches for dips and other candidate anomalies.

## Install from this checkout

The package is validated on Python 3.12:

```bash
python -m pip install -e ".[data,docs,test]"
```

See [installation](docs/user/install.md) for CPU-only PyTorch and environment instructions.
Version 0.2.0 is prepared locally. These instructions install the source checkout
without requiring a PyPI release.

## A minimal workflow

```python
from nahiku import Nahiku

lc = Nahiku.from_synthetic_noise(n_points=200, period=2,
                                white_noise=0.02, red_noise=0, seed=48)
lc.inject_anomaly(center=5, width=0.12, depth=0.8)
lc.standardize().fit_gp(period=2, length_scale=4, iterations=120)
result = lc.greedy_search(sigma=3)
fig, axes = result.plot()
print(lc.evaluate(result))
```

## Learn and reproduce

Download observations through the Lightkurve interface:

```python
lc = Nahiku.download("KIC 8462852", mission="Kepler", author="Kepler",
                     exptime=1800, quarter=16)
combined = Nahiku.download("KIC 8462852", mission="Kepler", author="Kepler",
                           exptime=1800, quarter=[15, 16], stitch=True)
```

`stitch=True` downloads all matches and median-normalizes each segment before
combining them. The default requires one product. See [loading details](docs/user/quickstart.md).

- [Synthetic light curve](examples/synthetic_lightcurve.ipynb): both generators, injections, fitting, searches, recovery.
- [Tabby's Star](examples/tabbys_star.ipynb): Lightkurve/Kepler loading and both searches.
- [β Pictoris](examples/beta_pic.ipynb): Lightkurve/TESS loading, prewhitening, and both searches.
- [User Guide](docs/index.md), [feature tutorials](docs/tutorials/index.md), and [API reference](docs/api/index.md).
- [Contributing](CONTRIBUTING.md), [citation metadata](CITATION.cff), and [JOSS readiness](JOSS_CHECKLIST.md).

Build the local documentation with `sphinx-build -W -b html docs docs/_build/html`.
Open `docs/_build/html/index.html`. Read the Docs configuration is included; hosting
requires connecting the repository to the service. Run `python -m pytest` to validate the installed package.

Detections depend on the statistical model and require further evidence for
physical classification.
Read the [statistical assumptions](docs/user/concepts.md). Maintained examples live in
`examples/`, numerical tests in `tests/`, and package configuration in `pyproject.toml`.

MIT licensed; see [third-party attribution](THIRD_PARTY_NOTICES.md).
