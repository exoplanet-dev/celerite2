#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sphinx_material
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("celerite2").version
except DistributionNotFound:
    __version__ = "unknown version"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

autodoc_mock_imports = [
    "numpy",
    "theano",
    "celerite2.driver",
    "celerite2.backprop",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "celerite2"
author = "Dan Foreman-Mackey"
copyright = "2020, " + author
version = __version__
release = __version__

exclude_patterns = ["_build"]

# HTML theme
html_show_sourcelink = False
html_sidebars = {
    "**": [
        "logo-text.html",
        "globaltoc.html",
        "localtoc.html",
        "searchbox.html",
    ]
}

extensions.append("sphinx_material")
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"
html_title = html_short_title = "celerite2"

html_favicon = "_static/logo.png"
html_logo = "_static/logo-inv.png"
html_theme_options = {
    "nav_title": "celerite2",
    "color_primary": "blue",
    "color_accent": "light-blue",
    "repo_url": "https://github.com/dfm/celerite2",
    "repo_name": "celerite2",
    "globaltoc_depth": 1,
    "globaltoc_collapse": False,
    "globaltoc_includehidden": False,
    "heroes": {
        "index": r"<strong>celerite</strong> &nbsp; &nbsp; \se.le.ʁi.te\ "
        r"&nbsp; &nbsp; <i>noun, archaic literary</i><br>"
        r"A scalable method for Gaussian Process regression in one dimension. "
        r"From French <i>célérité</i>."
    },
    "nav_links": [],
}
