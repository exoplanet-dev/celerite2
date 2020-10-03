# -*- coding: utf-8 -*-

__all__ = ["CeleriteNormal"]
import numpy as np
from theano import tensor as tt

try:
    import pymc3  # noqa
except ImportError:
    Continuous = object
    HAS_PYMC3 = False
else:
    from pymc3.distributions.distribution import Continuous, draw_values

    HAS_PYMC3 = True

from .. import driver


class CeleriteNormal(Continuous):
    """A multivariate normal distribution with a celerite covariance matrix

    Args:
        gp (celerite2.theano.GaussianProcess): The Gaussian Process computation
            engine.
    """

    def __init__(self, gp, *args, **kwargs):
        if not HAS_PYMC3:
            raise ImportError(
                "pymc3 is required to use the CeleriteNormal distribution"
            )

        super().__init__(*args, **kwargs)
        self.gp = gp
        self.mean = (
            self.median
        ) = self.mode = self.gp.mean_value + tt.zeros_like(self.gp._t)

    def random(self, point=None, size=None):
        if size is None:
            size = tuple()
        else:
            if not isinstance(size, tuple):
                try:
                    size = tuple(size)
                except TypeError:
                    size = (size,)

        mu, U, P, d, W = draw_values(
            [self.mean, self.gp._U, self.gp._P, self.gp._d, self.gp._W],
            point=point,
            size=size,
        )
        n = np.random.randn(*(size + tuple([d.shape[-1]])))

        func = np.vectorize(
            driver.dot_tril, signature="(n,j),(m,j),(n),(n,j),(n)->(n)"
        )
        return func(U, P, d, W, n) + mu

    def logp(self, y):
        return self.gp.log_likelihood(y)
