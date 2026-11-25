from pathlib import Path
import os
import nahiku
project = "Nāhiku"
author = "Linnea Wolniewicz"
copyright = "2026, Linnea Wolniewicz"
release = nahiku.__version__
extensions = ["myst_nb", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/linneawolniewicz/nahiku",
    "logo": {"text": "Nāhiku"},
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Nāhiku — stellar light-curve anomaly detection"
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "examples/data/**"]
nb_execution_mode = "off"  # CI separately executes the canonical notebooks before building.
nb_execution_timeout = 600
myst_enable_extensions = ["colon_fence", "dollarmath"]
nb_output_stderr = "show"
autodoc_member_order = "bysource"
