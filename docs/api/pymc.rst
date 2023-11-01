.. _pymc4-api:

PyMC (v5+) interface
===============

This ``celerite2.pymc`` submodule provides access to the *celerite2* models
within the `PyTensor <https://pytensor.readthedocs.io/>`_ framework. Of special
interest, this adds support for probabilistic model building using `PyMC
<https://docs.pymc.io/>`_ v5 or later.

*Note: PyMC v4 was a short-lived version of PyMC with the aesara backend.
Upgrading to PyMC 5 or above is the recommended way forward, but past releases of
celerite2 might work with aesara.*

The :ref:`first` tutorial demonstrates the use of this interface, while this
page provides the details for the :class:`celerite2.pymc4.GaussianProcess` class
which provides all this functionality. This page does not include documentation
for the term models defined in Aesara, but you can refer to the
:ref:`python-terms` section of the :ref:`python-api` documentation. All of those
models are implemented in Aesara and you can access them using something like
the following:

.. code-block:: python

   import aesara.tensor as at
   from celerite2.pymc4 import GaussianProcess, terms

   term = terms.SHOTerm(S0=at.scalar(), w0=at.scalar(), Q=at.scalar())
   gp = GaussianProcess(term)

The :class:`celerite2.pymc4.GaussianProcess` class is detailed below:

.. autoclass:: celerite2.pymc4.GaussianProcess
   :inherited-members:
   :exclude-members: sample, sample_conditional, recompute


PyMC (v4) support
-----------------

This implementation comes with a custom PyMC ``Distribution`` that represents a
multivariate normal with a *celerite* covariance matrix. This is used by the
:func:`celerite2.pymc4.GaussianProcess.marginal` method documented above which
adds a marginal likelihood node to a PyMC model.

.. autoclass:: celerite2.pymc4.distribution.CeleriteNormal
