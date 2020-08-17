.. _theano-api:

Theano interface
================

This ``celerite2.theano`` submodule provides access to the *celerite2* models
within the `Theano <http://deeplearning.net/software/theano/>`_ framework. Of
special interest, this adds support for probabilistic model building using
`PyMC3 <https://docs.pymc.io/>`_.

The TBD tutorial demonstrates the use of this interface, while this page
provides the details for the :class:`theano.GaussianProcess` class which
provides all this functionality. This page does not include documentation for
the term models defined in Theano, but you can refer to the :ref:`python-terms`
section of the :ref:`python-api` documentation. All of those models are
implemented in Theano and you can access them using something like the
following:

.. code-block:: python

   import theano.tensor as tt
   import celerite2.theano
   from celerite2.theano import terms

   term = terms.SHOTerm(S0=tt.dscalar(), w0=tt.dscalar(), Q=tt.dscalar())
   gp = celerite2.theano.GaussianProcess(term)

The :class:`celerite2.theano.GaussianProcess` class is detailed below:

.. autoclass:: celerite2.theano.GaussianProcess
   :inherited-members:
   :exclude-members: sample, sample_conditional, recompute


PyMC3 support
-------------

This implementation comes with a custom PyMC3 ``Distribution`` that represents a
multivariate normal with a *celerite* covariance matrix. This is used by the
:func:`celerite2.theano.GaussianProcess.marginal` method documented above which
adds a marginal likelihood node to a PyMC3 model.

.. autoclass:: celerite2.theano.CeleriteNormal
