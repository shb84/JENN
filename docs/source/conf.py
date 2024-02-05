"""Sphinx configuration file."""
import tomllib
from pathlib import Path

SRC_DOCS = Path(__file__).parent
DOCS = SRC_DOCS.parent 
ROOT = DOCS.parent
SRC = ROOT / "src"
PPT = ROOT / "pyproject.toml"
PPT_DATA = tomllib.loads(PPT.read_text(encoding="utf-8"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = PPT_DATA["project"]["name"]
authors = [author["name"] for author in PPT_DATA["project"]["authors"]]
release =  PPT_DATA["project"]["version"]
copyright = '2018, Steven H. Berguin'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    "sphinx_multiversion",
]

templates_path = ['_templates']
exclude_patterns = []

html_sidebars = {
    '**': [
        'versioning.html',
    ],
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

