.. _jax-api:

JAX interface
=============

This ``celerite2.jax`` submodule provides an interface to *celerite2* models
that can be used from `JAX <https://jax.readthedocs.io>`_.

The :ref:`first` tutorial demonstrates the use of this interface, while this
page provides the details for the :class:`celerite2.jax.GaussianProcess` class
which provides all this functionality. This page does not include documentation
for the term models defined in JAX, but you can refer to the
:ref:`python-terms` section of the :ref:`python-api` documentation. All of those
models are implemented in JAX and you can access them using something like
the following:

.. code-block:: python

   import jax
   from celerite2.jax import GaussianProcess, terms

   @jax.jit
   def log_likelihood(params, x, diag, y):
       term = terms.SHOTerm(S0=params["S0"], w0=params["w0"], Q=params["Q"])
       gp = GaussianProcess(term)
       gp.compute(x, diag=diag)
       return gp.log_likelihood(y)

The :class:`celerite2.jax.GaussianProcess` class is detailed below:

.. autoclass:: celerite2.jax.GaussianProcess
   :inherited-members:
   :exclude-members: sample, sample_conditional, recompute


numpyro support
---------------

This implementation comes with a custom `numpyro <https://num.pyro.ai>`_
``Distribution`` that represents a multivariate normal with a *celerite*
covariance matrix. This is used by the
:func:`celerite2.jax.GaussianProcess.numpyro_dist` method documented above which
adds a marginal likelihood node to a numpyro model.

.. autoclass:: celerite2.jax.distribution.CeleriteNormal
