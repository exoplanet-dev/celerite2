# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
from jax import numpy as np

from .. import backprop, driver
from ..ext import BaseGaussianProcess
from . import distribution, ops


class GaussianProcess(BaseGaussianProcess):
    def as_tensor(self, tensor):
        return np.asarray(tensor, dtype=np.float64)

    def zeros_like(self, tensor):
        return np.zeros_like(tensor)

    def do_compute(self, quiet):
        # Compute the Cholesky factorization
        try:
            self._d, self._W = ops.factor(self._a, self._U, self._V, self._P)
        except driver.LinAlgError:
            if not quiet:
                raise backprop.LinAlgError(
                    "failed to factorize or solve matrix"
                )
            self._log_det = -np.inf
            self._norm = np.inf
        else:
            self._log_det = np.sum(np.log(self._d))
            self._norm = -0.5 * (
                self._log_det + len(self._t) * np.log(2 * np.pi)
            )

    def check_sorted(self, t):
        return t

    def do_solve(self, y):
        return ops.solve(self._U, self._P, self._d, self._W, y)

    def do_dot_tril(self, y):
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)

    def do_norm(self, y):
        return ops.norm(self._U, self._P, self._d, self._W, y)

    def do_conditional_mean(self, *args):
        return ops.conditional_mean(self._U, self._V, self._P, *args)

    def tensordot(self, a, b):
        return np.tensordot(a, b, axes=(0, 0))

    def diagdot(self, a, b):
        return np.einsum("ij,ij->j", a, b)

    def numpyro_dist(self):
        return distribution.CeleriteNormal(self)
