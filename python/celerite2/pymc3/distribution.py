# -*- coding: utf-8 -*-

__all__ = ["CeleriteNormal"]

import numpy as np
import theano.tensor as tt
from celerite2 import driver

try:
    import pymc3 as pm
except ImportError:
    HAS_PYMC = False
    pm = None
    Continuous = object
else:
    HAS_PYMC = True
    Continuous = pm.distributions.distribution.Continuous


class CeleriteNormal(Continuous):
    """A multivariate normal distribution with a celerite covariance matrix

    Args:
        gp (celerite2.theano.GaussianProcess): The Gaussian Process computation
            engine.
    """

    def __init__(self, gp, *args, **kwargs):
        if not HAS_PYMC:
            raise ImportError(
                "PyMC is required to use the CeleriteNormal distribution"
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

        mu, U, P, d, W = pm.distributions.distribution.draw_values(
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
