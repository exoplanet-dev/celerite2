# celerite2

_celerite_ is an algorithm for fast and scalable Gaussian Process (GP)
Regression in one dimension and this library, _celerite2_ is a re-write of the
original [celerite project](https://celerite.readthedocs.io) to improve
numerical stability and integration with various machine learning frameworks.
This implementation includes interfaces in Python and C++, with full support for
Theano/PyMC3 and preliminary interfaces in JAX, PyTorch, and TensorFlow.

This documentation won't teach you the fundamentals of GP modeling but the best
resource for learning about this is available for free online: [Rasmussen &
Williams (2006)](http://www.gaussianprocess.org/gpml/). Similarly, the
_celerite_ algorithm is restricted to a specific class of covariance functions
(see [the original paper](https://arxiv.org/abs/1703.09710) for more information
and [a recent generalization](https://arxiv.org/abs/2007.05799) for extensions
to structured two-dimensional data). If you need scalable GPs with more general
covariance functions, [GPyTorch](https://gpytorch.ai/) might be a good choice.
