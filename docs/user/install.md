# Installation

Install Nāhiku from this source checkout. The package requires Python 3.12 or later
and is validated on Python 3.12. These instructions apply to the source distribution
and do not require a published PyPI release.

From the repository root, create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[data,docs,test]"
```

On Windows, activate with `.venv\Scripts\activate`. Core dependencies are NumPy,
SciPy, Astropy, Matplotlib, PyTorch, GPyTorch, and tqdm. `data` adds Lightkurve and its
archive tools; `docs` adds notebook and Sphinx tooling; `test` adds pytest.
`dev` combines these tools and build/lint dependencies.

For CPU-only PyTorch on Linux, install it first from the official CPU index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[data,docs,test]"
```

CUDA must be selected explicitly. PyPI's Linux PyTorch distribution may
include accelerator dependencies; the explicit CPU command avoids those downloads.
For a supported CUDA installation follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

For a locked developer environment use `uv sync --extra dev`; its PyTorch source is the CPU index on Linux/Windows and PyPI on macOS. `uv.lock` records the
resolved dependency set; compatibility still requires testing.
Use `pyproject.toml` and `uv.lock` to configure the supported environment.

Verify the import from a directory outside the checkout:

```bash
python -c "from nahiku import Nahiku, __version__; print(__version__)"
```

Run `python -m pytest` from the repository root to discover the package tests.
To launch notebooks, use the environment's Python kernel in Jupyter or your IDE.
