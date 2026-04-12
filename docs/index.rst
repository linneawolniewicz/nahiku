.. nahiku documentation master file, created by
   sphinx-quickstart on Sun Apr 12 15:56:08 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nāhiku: Anomaly Detection in Stellar Light Curves
=================================================

Nāhiku is the ʻŌlelo Hawaiʻi word for the 'Big Dipper' constellation.

This is an open-source package designed to simplify the detection of anomalies in stellar light curves using a principled, probabilistic approach. It provides tools for light curve modeling, prewhitening, and anomaly detection—specifically optimized for identifying "dipper" events, such as exocomets.

Features
--------

- **Principled Modeling**: Uses Gaussian Processes (GP) to model stellar variability.
- **Prewhitening**: Built-in support for removing stellar pulsations using the Balmung algorithm.
- **Greedy Search**: Fast iterative anomaly detection.
- **Exhaustive Search**: Comprehensive search for anomalies across varying window sizes.
- **Synthetic Data**: Tools to generate synthetic light curves with injected anomalies for testing and validation.

Installation
------------

You can install `nahiku` directly from PyPI:

.. code-block:: bash

   pip install nahiku

Quick Example
-------------

Here is a simple example of how to use Nāhiku to detect an injected anomaly in a synthetic light curve:

.. code-block:: python

   import numpy as np
   from nahiku import Nahiku

   # 1. Create a synthetic light curve
   x = np.arange(0, 100, 0.1)
   y = np.sin(x) + np.random.normal(0, 0.1, size=x.shape)

   # 2. Initialize Nahiku and prewhiten the signal
   nahiku = Nahiku(x, y)
   nahiku.prewhiten(minimum_snr=1)

   # 3. Inject an exocomet-shaped anomaly
   nahiku.inject_anomaly(1, absolute_width=0.5, absolute_depth=5, shapes=["exocomet"], idxs=[350])

   # 4. Search for the anomaly using Greedy Search
   res = nahiku.greedy_search()
   print(f"Found {res.num_detected_anomalies} anomalies.")

   # 5. Visualize the results
   nahiku.plot()

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
