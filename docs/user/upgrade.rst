.. _upgrade:

Upgrading from celerite
=======================

This package, *celerite2*, is a complete rewrite of the `celerite
<https://celerite.readthedocs.io>`_ library. While the interface is
qualitatively similar, the new implementation is not backwards compatible and
this page provides some information about how to update existing code.

Why does this even exist?
-------------------------

First, let's say a few words about why *celerite2* exists and why you might want
to use it. For the average user, the experience shouldn't be very different (the
computational speed is essentially identical), and in some ways the new package
has fewer features. In particular, all of the `model building features
<https://celerite.readthedocs.io/en/stable/tutorials/modeling/>`_ have been
removed. These have been removed because they were always a kludge and it
doesn't really make sense for a model building framework to be implemented
within a simple Gaussian Process library.

Since the original package was released (and before too), there have been many
interesting developments in the context of probabilistic model building driven,
in many cases, by excitement about deep learning. And there are now many Python
libraries that implement the features needed to build the probabilistic models
such as `PyMC3 <https://docs.pymc.io/>`_ (built on `Theano
<http://deeplearning.net/software/theano/>`_), `TensorFlow Probability
<https://www.tensorflow.org/probability>`_, and `Pyro <https://pyro.ai/>`_ (built
on `PyTorch <https://pytorch.org/>`_), to name only a few. Unfortunately the
original ``celerite`` package was not compatible with any of these frameworks and
a rewrite was required to provide consistent support across a range of use cases.

What interfaces are available in celerite2?
-------------------------------------------

At this point, *celerite2* includes:

- A mature and production-tested interface implemented in `Theano
  <http://deeplearning.net/software/theano/>`_, providing compatibility with
  `PyMC3 <https://docs.pymc.io/>`_. This interface was originally implemented as
  part of the `exoplanet <https://docs.exoplanet.codes>`_ project, and it is the
  recommended interface for most use cases.

- A simple pure-Python interface that can be used for evaluating *celerite*
  models that is suitable for use with inference tools like
  :func:`scipy.optimize.minimize`, `emcee <https://emcee.readthedocs.io>`_, and
  `dynesty <https://dynesty.readthedocs.io/>`_. Note, however that in all of
  these cases, users will be responsible for defining their own models.

- Experimental interfaces implemented in `JAX <https://github.com/google/jax>`_,
  `PyTorch <https://pytorch.org/>`_, and `TensorFlow
  <https://www.tensorflow.org/probability>`_. These interfaces are not as
  optimized or as well documented as the other two, but that should change in
  the future.

A simple example
----------------

If you're starting on a new project, it would be best to start with the
tutorials, but this section provides a simple example of how the syntax for
*celerite2* compares to the original *celerite* package.

Let's say that we have the following model defined using *celerite*:

.. code-block:: python

    import numpy as np

    import celerite
    from celerite import terms

    x, y, yerr = data

    kernel = terms.SHOTerm(log_S0=np.log(1.0), log_Q=np.log(2.5), log_omega0=np.log(0.2))
    gp = celerite.GP(kernel, mean=2.123)
    gp.compute(x, yerr)
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))

You can define the equivalent model in *celerite2* as follows:

.. code-block:: python

    import numpy as np

    import celerite2
    from celerite2 import terms

    x, y, yerr = data

    kernel = terms.SHOTerm(S0=1.0, Q=2.5, w0=0.2)
    gp = celerite2.GaussianProcess(kernel, mean=2.123)
    gp.compute(x, yerr=yerr)
    print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))

While this looks quite similar, the major differences include:

- the term parameters are no longer logarithmic,
- the computation class is called :class:`celerite2.GaussianProcess` instead of
  ``GP``, and
- ``yerr`` must now be passed as a keyword.

To find the maximum likelihood model using ``scipy``, you would execute
something like the following:

.. code-block:: python

    from scipy.optimize import minimize

    def set_params(params, gp):
        gp.mean = params[0]
        theta = np.exp(params[1:])
        gp.kernel = terms.SHOTerm(S0=theta[0], Q=theta[1], w0=theta[2])
        return gp

    def neg_log_like(params, gp):
        gp = set_params(params, gp)
        gp.recompute(quiet=True)
        return -gp.log_likelihood(y)

    initial_params = [2.123, np.log(kernel.S0), np.log(kernel.Q), np.log(kernel.w0)]
    soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
    opt_gp = set_params(soln.x, gp)

This is somewhat more verbose than the equivalent operation using *celerite* and
it doesn't include all the niceties like built in parameter bounds, but it
wouldn't be too much to implement these for a specific use case.
