#!/usr/bin/env python
# -*- coding: utf-8 -*-
import subprocess

import sphinx_material
from pkg_resources import DistributionNotFound, get_distribution

try:
    __version__ = get_distribution("celerite2").version
except DistributionNotFound:
    __version__ = "dev"


subprocess.call("doxygen Doxyfile.in", shell=True)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_nb",
    "breathe",
]

autodoc_mock_imports = ["theano", "pymc3"]

exclude_patterns = ["_build"]
templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

# General information about the project.
project = "celerite2"
author = "Dan Foreman-Mackey"
copyright = "2020-2022 " + author
version = __version__
release = __version__

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

# Theme
extensions.append("sphinx_material")
html_theme_path = sphinx_material.html_theme_path()
html_context = sphinx_material.get_html_context()
html_theme = "sphinx_material"
html_title = html_short_title = "celerite2"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

html_favicon = "_static/logo.png"
html_logo = "_static/logo-inv.png"
html_theme_options = {
    "nav_title": "celerite2",
    "color_primary": "blue",
    "color_accent": "light-blue",
    "repo_url": "https://github.com/exoplanet-dev/celerite2",
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

# Breathe for C++
breathe_projects = {"celerite2": "c++/xml/"}
breathe_default_project = "celerite2"
breathe_default_members = ("members", "undoc-members")

# Myst notebooks
nb_execution_mode = "auto"
nb_execution_timeout = -1
