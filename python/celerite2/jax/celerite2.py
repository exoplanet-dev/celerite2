# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
from jax import numpy as np

from .. import backprop, driver
from ..celerite2 import ConstantMean
from ..celerite2 import GaussianProcess as SuperGaussianProcess
from . import ops


class GaussianProcess(SuperGaussianProcess):
    def __init__(self, kernel, t=None, *, mean=0.0, **kwargs):
        self._kernel = kernel
        if callable(mean):
            self._mean = mean
        else:
            self._mean = ConstantMean(mean)

        # Placeholders for storing data
        self._t = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        if t is not None:
            self.compute(t, **kwargs)

    def compute(
        self, t, *, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        # Check the input coordinates
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if len(t.shape) != 1:
            raise ValueError("the input coordinates must be one dimensional")

        # Save the diagonal
        self._t = np.asarray(t, dtype=np.float64)
        self._diag = np.zeros_like(self._t)
        if yerr is None and diag is None:
            pass

        elif yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag += np.atleast_1d(yerr) ** 2

        else:
            self._diag += np.atleast_1d(diag)

        # Fill the celerite matrices
        a, self._U, self._V, self._P = self._kernel.get_celerite_matrices(
            self._t, self._diag
        )

        # Compute the Cholesky factorization
        try:
            self._d, self._W = ops.factor(a, self._U, self._V, self._P)
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

    def recompute(self, *, quiet=False):
        if self._t is None:
            raise RuntimeError(
                "you must call 'compute' directly  at least once"
            )
        return self.compute(
            self._t, diag=self._diag, check_sorted=False, quiet=quiet
        )

    def _process_input(self, y, *, require_vector=False):
        y = np.atleast_1d(y)
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if self._t.shape[0] != y.shape[0]:
            raise ValueError("dimension mismatch")
        if require_vector and self._t.shape != y.shape:
            raise ValueError("'y' must be one dimensional")
        y = np.array(y, dtype=np.float64, copy=True)
        return y

    def apply_inverse(self, y):
        y = self._process_input(y)
        return ops.solve(self._U, self._P, self._d, self._W, y)

    def dot_tril(self, y):
        y = self._process_input(y)
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)

    def log_likelihood(self, y):
        y = self._process_input(y, require_vector=True)
        if not np.isfinite(self._log_det):
            return -np.inf
        loglike = self._norm - 0.5 * ops.norm(
            self._U, self._P, self._d, self._W, y - self._mean(self._t)
        )
        if not np.isfinite(loglike):
            return -np.inf
        return loglike

    def predict(
        self,
        y,
        t=None,
        *,
        return_cov=False,
        return_var=False,
        include_mean=True
    ):
        y = self._process_input(y, require_vector=True)

        alpha = ops.solve(
            self._U, self._P, self._d, self._W, y - self._mean(self._t)
        )

        if t is None:
            xs = self._t
            mu = y - self._diag * alpha

        else:
            xs = np.asarray(t, dtype=np.float64)
            if xs.ndim != 1:
                raise ValueError("dimension mismatch")

            U_star, V_star, inds = self._kernel.get_conditional_mean_matrices(
                self._t, xs
            )
            mu = ops.conditional_mean(
                self._U, self._V, self._P, alpha, U_star, V_star, inds
            )
            if include_mean:
                mu += self._mean(self._t)

        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        KxsT = self._kernel.get_value(xs[None, :] - self._t[:, None])
        if return_var:
            var = self._kernel.get_value(0.0) - np.einsum(
                "ij,ij->j", KxsT, self.apply_inverse(KxsT)
            )
            return mu, var

        # Predictive covariance
        cov = self._kernel.get_value(xs[:, None] - xs[None, :])
        cov -= np.tensordot(KxsT, self.apply_inverse(KxsT), axes=(0, 0))
        return mu, cov

    def sample(self, *args, **kwargs):
        raise NotImplementedError("'sample' is not implemented in jax")

    def sample_conditional(self, *args, **kwargs):
        raise NotImplementedError(
            "'sample_conditional' is not implemented in jax"
        )
