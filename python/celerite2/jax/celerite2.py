# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess", "ConditionalDistribution"]
from jax import numpy as np

from ..core import BaseConditionalDistribution, BaseGaussianProcess
from . import distribution, ops


class ConditionalDistribution(BaseConditionalDistribution):
    def _do_general_matmul(self, c, U1, V1, U2, V2, inp, target):
        target += ops.general_matmul_lower(
            self._xs, self.gp._t, c, U2, V1, inp
        )
        target += ops.general_matmul_upper(
            self._xs, self.gp._t, c, V2, U1, inp
        )
        return target

    def _diagdot(self, a, b):
        return np.einsum("ij,ij->j", a, b)


class GaussianProcess(BaseGaussianProcess):
    conditional_distribution = ConditionalDistribution

    def _as_tensor(self, tensor):
        return np.asarray(tensor, dtype=np.float64)

    def _zeros_like(self, tensor):
        return np.zeros_like(tensor)

    def _do_compute(self, quiet):
        self._d, self._W = ops.factor(
            self._t, self._c, self._a, self._U, self._V
        )
        self._log_det = np.sum(np.log(self._d))
        self._norm = -0.5 * (self._log_det + self._size * np.log(2 * np.pi))

    def _check_sorted(self, t):
        return t

    def _do_solve(self, y):
        z = ops.solve_lower(self._t, self._c, self._U, self._W, y)
        z /= self._d[:, None]
        z = ops.solve_upper(self._t, self._c, self._U, self._W, z)
        return z

    def _do_dot_tril(self, y):
        z = y * np.sqrt(self._d)[:, None]
        z += ops.matmul_lower(self._t, self._c, self._U, self._W, z)
        return z

    def _do_norm(self, y):
        alpha = ops.solve_lower(
            self._t, self._c, self._U, self._W, y[:, None]
        )[:, 0]
        return np.sum(alpha ** 2 / self._d)

    def numpyro_dist(self):
        return distribution.CeleriteNormal(self)
