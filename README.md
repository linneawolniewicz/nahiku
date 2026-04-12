# Nāhiku

**Anomaly Detection in Stellar Light Curves**

Nāhiku is the ʻŌlelo Hawaiʻi word for the 'Big Dipper' constellation.

This is an open-source package to simplify the task of detecting anomalies in stellar light curves with a principled, probabilistic approach to light curve modeling and anomaly detection. Specifically, this package was developed for the use of detecting dipper anomalies, such as exocomets, in light curves.

## Features

- **Light Curve Generation**: Support for loading and generating light curves from various sources such as Kepler / TESS, custom CSV files, and synthetic data generation. Synthetic light curves can be generated from user-specified noise parameters, or sampled from a multivariate normal distribution with a user-specified covariance kernel. 
- **Gaussian Process Modeling**: Principled modeling of stellar variability, with GPU-acceleration via GPyTorch.
- **Prewhitening**: Built-in support for removing stellar pulsations via the Balmung algorithm.
- **Anomaly Search**: Both **Greedy Search** (fast, iterative) and **Exhaustive Search** (comprehensive window search) methods.

## Installation

Nāhiku can be installed directly from PyPI:

```bash
pip install nahiku
```

## Quick Example

Here is a short script to create a synthetic light curve, inject an anomaly, and recover it using the Greedy Search method:

```python
import numpy as np
from nahiku import Nahiku

# 1. Create a simple synthetic light curve
x = np.arange(0, 100, 0.1)
y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

# 2. Initialize Nahiku object and prewhiten the signal
nahiku = Nahiku(x, y)
nahiku.prewhiten(minimum_snr=1)

# 3. Inject an exocomet-shaped anomaly
nahiku.inject_anomaly(1, absolute_width=0.5, absolute_depth=5, shapes=["exocomet"], idxs=[350])

# 4. Search for the anomaly using Greedy Search
res = nahiku.greedy_search()
print(f"Greedy search found {res.num_detected_anomalies} anomalies.")

# 5. Visualize the search process and results
nahiku.plot()
```

## Documentation

Full documentation, including API references and examples, can be found at: [https://nahiku.readthedocs.io/](https://nahiku.readthedocs.io/) (once hosted).

## License

This project is licensed under the MIT License.
