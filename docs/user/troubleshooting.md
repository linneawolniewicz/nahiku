# Troubleshooting

- **Import errors:** install this checkout using its `pyproject.toml`; do not add `src/nahiku`
  to `sys.path`. Check that the notebook kernel uses the same environment.
- **No accepted period:** inspect the spectrum and bounds; correlated noise or a boundary
  peak can invalidate a simple period interpretation. Supply a justified period or use RBF.
- **No detections:** inspect preprocessing, prediction uncertainty, window widths, and
  signal/noise. A search may return no detections. Selecting top-k candidates alone does not
  establish statistical significance.
- **Many detections:** residual variability, underestimated errors, or an inflexible kernel
  may violate the Gaussian model. Inspect the training loss and stopping reason:
  an iteration-limited fit can produce widespread detections. Check baseline
  flexibility and prediction uncertainty before tightening thresholds. A smaller
  exhaustive p-value cutoff is more restrictive. Test null simulations and parameter sensitivity.
- **Prewhitening reaches its cap:** inspect residual spectra and dip preservation before
  increasing the cap. Removing more frequencies can also remove signal from dips.
- **Archive unavailable/product checksum changed:** use `NAHIKU_OFFLINE=1` for the frozen
  examples. Review a changed archive product before updating its manifest.
- **Out of memory:** reduce the bounded analysis window or bin with a justified cadence.
  Dense covariance matrices for a full mission can exceed available memory.

For support, [open an issue](https://github.com/linneawolniewicz/nahiku/issues) with version,
OS, minimal code, traceback, and a small non-sensitive sample or public product identifier.
