# Changelog

## 0.2.0 — Unreleased

- Greedy search supports NumPy/SciPy and Torch conditioning backends with the same
  posterior-residual and leave-one-out seed statistics. Small-matrix validation compares conditioning and
  predictive uncertainties against GPyTorch exact posteriors in both precisions.

- Training defaults (1,000 steps, lr=0.01, patience=1) and explicit warm-up,
  stopping, MLL/MSE objective, last/best model parameter state, and float32/float64 controls.
- Optional residual-variance noise reset runs before inference-cache construction.
  Add optional final MLL return and `plot_fit()` diagnostics.

- Period-informed GP parameter bounds and initialization via `qp_constraints="period-informed"`.
- Standardize before GP fitting by default, with `standardize=False` to opt out.
- Reuse in-memory factors and bounded prediction/subset caches without disk serialization.

- Select prewhitening with `method="balmung"` (default) or `method="weighted"`.
  Both retain diagnostics and preserve the continuum. The package includes Balmung;
  its boundary-peak frequency bug is corrected. The beta Pic case study uses the weighted method.

- Optional `estimate_period(fallback="shoulder")` provides smoothed-spectrum
  timescale estimation with explicit diagnostics and an observation-duration cap.

- `download(stitch=True)` optionally combines all matching Lightkurve products,
  preserving per-product metadata; the default still requires a single product.

- Central-object workflow that preserves observations, uncertainties, and preprocessing history.
- Period estimation using observation timestamps, with retained prewhitening diagnostics.
- Independent greedy and exhaustive results, validation of direct and precision-based conditioning, and numerical exports.
- Three complete notebooks, thirteen feature tutorials, and Sphinx/Read the Docs configuration.
- Numerical regression tests using scientific dependencies, packaging checks, provenance manifests, and JOSS preparation.

