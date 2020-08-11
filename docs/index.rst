=========
celerite2
=========

*celerite* is an algorithm for fast and scalable Gaussian Process (GP)
Regression in one dimension and this library, *celerite2* is a re-write of the
original `celerite project <https://celerite.readthedocs.io>`_ to improve
numerical stability and integration with various machine learning frameworks.
This implementation includes interfaces in Python and C++, with full support for
Theano/PyMC3 and preliminary interfaces in JAX, PyTorch, and TensorFlow.

This documentation won't teach you the fundamentals of GP modeling but the best
resource for learning about this is available for free online: `Rasmussen &
Williams (2006) <http://www.gaussianprocess.org/gpml/>`_. Similarly, the
*celerite* algorithm is restricted to a specific class of covariance functions
(see `the original paper <https://arxiv.org/abs/1703.09710>`_ for more
information and `a recent generalization <https://arxiv.org/abs/2007.05799>`_
for extensions to structured two-dimensional data). If you need scalable GPs
with more general covariance functions, `GPyTorch <https://gpytorch.ai/>`_ might
be a good choice.

*celerite2* is being actively developed in `a public repository on GitHub
<https://github.com/dfm/celerite>`_ so if you have any trouble, `open an issue
<https://github.com/dfm/celerite/issues>`_ there.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   install
   api

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/first.ipynb

License & attribution
---------------------

Copyright 2020 Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite the relevant papers with the following
BibTeX:

.. code-block:: bib

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

Changelog
---------

.. include:: ../HISTORY.rst
