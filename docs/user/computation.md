# Execution and computational limits

All examples default to CPU float64. Set `device="cuda"` in `fit_gp` when a
CUDA-enabled PyTorch installation is available. GP training and Torch search
conditioning use the selected device. The NumPy/SciPy backend uses the CPU; scalar
interval scores are also evaluated on the CPU. Requesting unavailable CUDA raises an error.

Dense covariance matrices require O(N²) memory and O(N³) factorization work. Exhaustive
precision reuse reduces repeated full-matrix factorizations but still scores many windows.
Per-interval hyperparameter refitting is much slower. GPU speed and survey throughput require benchmarks on specified data and hardware.

For reproducible CPU examples, set thread counts before starting Python:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python your_analysis.py
```

The library does not change global thread limits or random seeds. Simulation methods
use local generators. Built-in GP initial values are deterministic. Training requests
Cholesky evaluation for matrices below 10,000 samples; larger problems may use
GPyTorch’s iterative or stochastic calculations. The `seed` fitting argument records
provenance but does not seed such calculations or user-supplied modules. Dense
NumPy/SciPy inference caches are constructed after fitting regardless of training size.
Floating-point results can still differ slightly across platforms and dependency versions.

The main notebooks select limited observation windows and bin after preprocessing. For long light
curves, split by observation and cadence gaps, then analyze overlapping bounded windows.
Check edge detections and reconcile duplicate events by time. Fitting an entire multi-year light curve as one dense matrix can exceed available
memory. The package does not implement automatic multiprocessing.
