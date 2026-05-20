import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "diffct_mlx"
copyright = "2026, Yipeng Sun, Linda-Sophie Schneider"
author = "Yipeng Sun, Linda-Sophie Schneider"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = ["mlx", "mlx.core", "mlx.optimizers"]
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
