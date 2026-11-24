"""Execute maintained notebooks; fail immediately on a scientific/runtime error."""

import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument(
    "--real", action="store_true", help="Run the two archive-backed examples instead"
)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
paths = (
    [root / "examples" / f"{name}.ipynb" for name in ("tabbys_star", "beta_pic")]
    if args.real
    else [
        root / "examples/synthetic_lightcurve.ipynb",
        *sorted((root / "docs/tutorials").glob("*.ipynb")),
    ]
)
for path in paths:
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    ).execute()
    nbformat.write(notebook, path)
    print("Executed", path.relative_to(root), flush=True)
