=========
celerite2
=========

*celerite* is an algorithm for fast and scalable Gaussian Process (GP)
Regression in one dimension and this library, *celerite2* is a re-write of the
original `celerite project <https://celerite.readthedocs.io>`_ to improve
numerical stability and integration with various machine learning frameworks.
This implementation includes interfaces in Python and C++, with full support for
Theano/PyMC3 and experimental interfaces in JAX, PyTorch, and TensorFlow.

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
<https://github.com/exoplanet-dev/celerite2>`_ so if you have any trouble, `open an issue
<https://github.com/exoplanet-dev/celerite2/issues>`_ there.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/install
   user/upgrade
   user/citation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/first.ipynb

.. toctree::
   :maxdepth: 2
   :caption: API Details

   api/python
   api/theano
   api/c++

License & attribution
---------------------

Copyright 2020 Daniel Foreman-Mackey.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite the relevant papers: :ref:`citation`

Changelog
---------

.. include:: ../HISTORY.rst
