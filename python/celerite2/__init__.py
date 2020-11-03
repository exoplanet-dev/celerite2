# -*- coding: utf-8 -*-

__all__ = ["__version__", "terms", "latent", "GaussianProcess"]
from . import latent, terms
from .celerite2 import GaussianProcess
from .celerite2_version import __version__

__uri__ = "https://celerite2.readthedocs.io"
__author__ = "Daniel Foreman-Mackey"
__email__ = "foreman.mackey@gmail.com"
__license__ = "MIT"
__description__ = "Fast and scalable Gaussian Processes in 1D"
__bibtex__ = __citation__ = r"""
@article{celerite2:foremanmackey17,
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
@article{celerite2:foremanmackey18,
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
@article{celerite2:gordon20,
       author = {{Gordon}, Tyler and {Agol}, Eric and
                 {Foreman-Mackey}, Daniel},
        title = "{A Fast, 2D Gaussian Process Method Based on Celerite:
                  Applications to Transiting Exoplanet Discovery and
                  Characterization}",
      journal = {arXiv e-prints},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.05799},
        pages = {arXiv:2007.05799},
archivePrefix = {arXiv},
       eprint = {2007.05799},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200705799G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
"""
