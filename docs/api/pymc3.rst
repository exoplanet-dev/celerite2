.. _pymc3-api:

PyMC3 interface
===============

This ``celerite2.pymc3`` submodule provides access to the *celerite2* models
within the `Theano <http://deeplearning.net/software/theano/>`_ framework. Of
special interest, this adds support for probabilistic model building using
`PyMC3 <https://docs.pymc.io/>`_.

This page does not include documentation for the term models defined in Theano,
but you can refer to the :ref:`python-terms` section of the :ref:`python-api`
documentation. All of those models are implemented in Theano and you can access
them using something like the following:

.. code-block:: python

   import theano.tensor as tt
   from celerite2.pymc3 import GaussianProcess, terms

   term = terms.SHOTerm(S0=tt.dscalar(), w0=tt.dscalar(), Q=tt.dscalar())
   gp = GaussianProcess(term)

The :class:`celerite2.pymc3.GaussianProcess` class is detailed below:

.. autoclass:: celerite2.pymc3.GaussianProcess
   :inherited-members:
   :exclude-members: sample, sample_conditional, recompute


PyMC3 support
-------------

This implementation comes with a custom PyMC3 ``Distribution`` that represents a
multivariate normal with a *celerite* covariance matrix. This is used by the
:func:`celerite2.pymc3.GaussianProcess.marginal` method documented above which
adds a marginal likelihood node to a PyMC3 model.

.. autoclass:: celerite2.pymc3.distribution.CeleriteNormal
