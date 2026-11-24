"""Nāhiku: reproducible Gaussian-process anomaly detection in stellar light curves."""

__version__ = "0.2.0"
from .nahiku import Nahiku
from .results import SearchResult
from .preprocessing import PrewhiteningResult
from .gp_helpers import QuasiPeriodicKernel

__all__ = ["Nahiku", "SearchResult", "PrewhiteningResult", "QuasiPeriodicKernel", "__version__"]
