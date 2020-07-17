# -*- coding: utf-8 -*-

__all__ = ["__version__", "terms", "GaussianProcess"]

from . import terms
from .celerite2 import GaussianProcess
from .celerite2_version import __version__

__uri__ = "https://celerite2.readthedocs.io"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "Fast and scalable Gaussian Processes in 1D"
__bibtex__ = __citation__ = r"""
@article{celerite1,
   author = {{Foreman-Mackey}, D. and {Agol}, E. and {Ambikasaran}, S. and
             {Angus}, R.},
    title = "{Fast and Scalable Gaussian Process Modeling with Applications to
              Astronomical Time Series}",
  journal = {\aj},
     year = 2017,
    month = dec,
   volume = 154,
    pages = {220},
      doi = {10.3847/1538-3881/aa9332},
   adsurl = {http://adsabs.harvard.edu/abs/2017AJ....154..220F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@article{celerite2,
   author = {{Foreman-Mackey}, D.},
    title = "{Scalable Backpropagation for Gaussian Processes using Celerite}",
  journal = {Research Notes of the American Astronomical Society},
     year = 2018,
    month = feb,
   volume = 2,
   number = 1,
    pages = {31},
      doi = {10.3847/2515-5172/aaaf6c},
   adsurl = {http://adsabs.harvard.edu/abs/2018RNAAS...2a..31F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
