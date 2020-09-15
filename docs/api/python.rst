.. _python-api:

Python interface
================

The primary interface to computations using *celerite2* are provided by the
:class:`celerite2.GaussianProcess` class that is documented below. These
calculations will be performed using :class:`Term` models as documented in the
:ref:`python-terms` section. See the TBD tutorial for examples of how to use
this interface.

.. autoclass:: celerite2.GaussianProcess
   :inherited-members:

.. _python-terms:

Model building
--------------

.. note:: If you're not sure which model to use, check out the
   :ref:`python-recommended` section and, in particular, the :class:`SHOTerm`.

The following is the abstract base class for all the models provided by
*celerite2* and all of these methods will be implemented by subclasses.
All of these models can be composed using addition and multiplication.
For example,

.. code-block:: python

   from celerite2 import terms

   term1 = terms.SHOTerm(sigma=1.0, w0=0.5, Q=2.5)
   term2 = terms.SHOTerm(sigma=0.5, w0=0.5, Q=0.2)

   term = term1 + term2

   # or ...

   term = term1 * term2

.. autoclass:: celerite2.terms.Term
   :inherited-members:

.. _python-recommended:

Recommended models
++++++++++++++++++

These are the best models to use for most datasets.

.. autoclass:: celerite2.terms.SHOTerm
.. autoclass:: celerite2.terms.RotationTerm
.. autoclass:: celerite2.terms.Matern32Term

Operations
++++++++++

These classes encapsulate various operations on terms.

.. module:: celerite2.terms
.. autoclass:: celerite2.terms.TermSum
.. autoclass:: celerite2.terms.TermProduct
.. autoclass:: celerite2.terms.TermDiff
.. autoclass:: celerite2.terms.TermConvolution

Other models
++++++++++++

These models are included for backwards compatibility, but their use is not
recommended unless you're confident that you know what you're doing.

.. autoclass:: celerite2.terms.RealTerm
.. autoclass:: celerite2.terms.ComplexTerm
.. autoclass:: celerite2.terms.OriginalCeleriteTerm
