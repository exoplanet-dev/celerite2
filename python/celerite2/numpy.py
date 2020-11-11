# -*- coding: utf-8 -*-

__all__ = ["ConditionalDistribution", "GaussianProcess"]
import warnings

import numpy as np

from . import driver
from .core import BaseConditionalDistribution, BaseGaussianProcess
from .driver import LinAlgError


class ConditionalDistribution(BaseConditionalDistribution):
    def _do_general_matmul(self, c, U1, V1, U2, V2, inp, target):
        target = driver.general_matmul_lower(
            self._xs, self.gp._t, c, U2, V1, inp, target
        )
        target = driver.general_matmul_upper(
            self._xs, self.gp._t, c, V2, U1, inp, target
        )
        return target

    def _diagdot(self, a, b):
        return np.einsum("ij,ij->j", a, b)

    def sample(self, *, size=None, regularize=None):
        mu = self.mean
        cov = self.covariance
        if regularize is not None:
            cov[np.diag_indices_from(cov)] += regularize
        return np.random.multivariate_normal(mu, cov, size=size)


class GaussianProcess(BaseGaussianProcess):
    conditional_distribution = ConditionalDistribution

    def _setup(self):
        self._c = np.empty(0, dtype=np.float64)
        self._a = np.empty(0, dtype=np.float64)
        self._U = np.empty((0, 0), dtype=np.float64)
        self._V = np.empty((0, 0), dtype=np.float64)

    def _copy_or_check(self, y, *, inplace=False):
        if inplace:
            if (
                y.dtype != "float64"
                or not y.flags.c_contiguous
                or not y.flags.writeable
            ):
                warnings.warn(
                    "Inplace operations can only be made on C-contiguous, "
                    "writable, float64 arrays; a copy will be made"
                )
            y = np.ascontiguousarray(y, dtype=np.float64)
        else:
            y = np.array(y, dtype=np.float64, copy=True, order="C")
        return y

    def _as_tensor(self, y):
        return np.ascontiguousarray(y, dtype=np.float64)

    def _zeros_like(self, y):
        return np.zeros_like(y)

    def _do_compute(self, quiet):
        # Compute the Cholesky factorization
        try:
            self._d, self._W = driver.factor(
                self._t,
                self._c,
                self._a,
                self._U,
                self._V,
                self._a,
                np.copy(self._V),
            )
        except LinAlgError:
            if not quiet:
                raise
            self._log_det = -np.inf
            self._norm = np.inf
        else:
            self._log_det = np.sum(np.log(self._d))
            self._norm = -0.5 * (
                self._log_det + self._size * np.log(2 * np.pi)
            )

    def _check_sorted(self, t):
        if np.any(np.diff(t) < 0.0):
            raise ValueError("The input coordinates must be sorted")
        return t

    def _do_solve(self, y):
        z = driver.solve_lower(self._t, self._c, self._U, self._W, y, y)
        z /= self._d[:, None]
        z = driver.solve_upper(self._t, self._c, self._U, self._W, z, z)
        return z

    def _do_dot_tril(self, y):
        z = y * np.sqrt(self._d)[:, None]
        return driver.matmul_lower(self._t, self._c, self._U, self._W, z, z)

    def _do_norm(self, y):
        alpha = y[:, None]
        alpha = driver.solve_lower(
            self._t, self._c, self._U, self._W, alpha, alpha
        )[:, 0]
        return np.sum(alpha ** 2 / self._d)

    def sample(self, *, size=None, include_mean=True):
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if size is None:
            n = np.random.randn(self._size)
        else:
            n = np.random.randn(self._size, size)
        result = self.dot_tril(n, inplace=True).T
        if include_mean:
            result += self._mean_value
        return result
