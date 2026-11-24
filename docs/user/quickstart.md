# Quickstart

Generate a synthetic light curve, inject a dip, and search for it:

```python
from nahiku import Nahiku

lc = Nahiku.from_synthetic_noise(n_points=200, period=2,
                                white_noise=0.02, red_noise=0, seed=48)
lc.inject_anomaly(center=5, width=0.12, depth=0.8)
lc.standardize()
lc.fit_gp(period=2, length_scale=4, iterations=120)
greedy = lc.greedy_search(sigma=3)
exhaustive = lc.exhaustive_search(min_points=3, max_points=16,
                                step=3, size_step=3, threshold=1e-8)
fig, axes = greedy.plot()
print(lc.evaluate(greedy))
```

`Nahiku(time, flux, flux_err=...)` accepts your own arrays. Construction leaves flux
unchanged. Preprocessing methods are explicit, but `fit_gp()` standardizes by default unless
working flux already has zero mean and unit standard deviation. Use
`standardize=False` to opt out; `original_flux` and `original_flux_err` remain
available. `results["greedy"]` and `results["exhaustive"]` are separate snapshots.

With the `data` extra installed, Nāhiku can query the archive through Lightkurve.
Load one product or stitch multiple quarters:

```python
single = Nahiku.download("KIC 8462852", mission="Kepler", author="Kepler",
                         exptime=1800, quarter=16)
combined = Nahiku.download("KIC 8462852", mission="Kepler", author="Kepler",
                           exptime=1800, quarter=[15, 16], stitch=True)
```

The default `stitch=False` requires exactly one match. With `stitch=True`, all
matching products are downloaded, each segment is median-normalized by Lightkurve,
and the segments are concatenated. Omitting `quarter`/`sector` selects all matching
epochs. `quality_bitmask` and `cache_dir` apply to all downloads. Conversion then
filters invalid samples and sorts timestamps; observing gaps remain, and duplicate
timestamps raise an error. Stitching does not flatten or detrend variability.
Per-product metadata and row ranges are retained in `metadata["source_products"]`.

See the [Lightkurve search documentation](https://lightkurve.github.io/lightkurve/reference/api/lightkurve.search_lightcurve.html)
for the selection parameters. Large stitched datasets require dense GP matrices;
select an analysis window or bin the observations before fitting.

If you already have a Lightkurve object, convert it directly:

```python
import lightkurve as lk
observations = lk.search_lightcurve("KIC 8462852", mission="Kepler",
                                   author="Kepler", exptime=1800, quarter=16)
lightcurve = observations.download(quality_bitmask="default")
lc = Nahiku.from_lightkurve(lightcurve)
```

The main [examples](../examples/index.md) specify fixed archive products, quality masks, normalization,
analysis windows, and reproducible settings.
