# -*- coding: utf-8 -*-

__all__ = ["CeleriteNormal"]

import numpy as np
import pymc3 as pm
import theano.tensor as tt
from pymc3.distributions.distribution import Continuous

import celerite2.driver


class CeleriteNormal(Continuous):
    """A multivariate normal distribution with a celerite covariance matrix

    Args:
        gp (celerite2.theano.GaussianProcess): The Gaussian Process computation
            engine.
    """

    def __init__(self, gp, *args, **kwargs):
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
            celerite2.driver.dot_tril,
            signature="(n,j),(m,j),(n),(n,j),(n)->(n)",
        )
        return func(U, P, d, W, n) + mu

    def logp(self, y):
        return self.gp.log_likelihood(y)
