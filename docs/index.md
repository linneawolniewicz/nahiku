# Nāhiku

Nāhiku detects anomalous intervals in stellar light curves using Gaussian-process
models of background variability. It supports studies of dips, including exocomet candidates, and tests of detection
methods on controlled synthetic observations. Nāhiku is the ʻŌlelo Hawaiʻi name for the Big Dipper.

Start with the [quickstart](user/quickstart.md), then run one of the
[complete examples](examples/index.md). The [tutorials](tutorials/index.md) explain
each step separately. The [API reference](api/index.md) documents the supported interface.

The package combines explicit preprocessing, quasi-periodic or smooth GP modeling,
greedy detection, and exhaustive conditional interval scoring. Detections are statistical candidates. Identifying an exocomet requires further
astrophysical evidence.

```{toctree}
:caption: User Guide
:maxdepth: 1

user/install
user/quickstart
user/concepts
user/citations
user/computation
user/troubleshooting
api/index
developer/index
changes
```

```{toctree}
:caption: Tutorials and examples
:maxdepth: 1

tutorials/index
examples/index
```
