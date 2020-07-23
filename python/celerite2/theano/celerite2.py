# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
import numpy as np
from theano import tensor as tt

from . import ops


class GaussianProcess:
    def __init__(self, kernel, t=None, diag=None):
        self._kernel = kernel

        # Placeholders for storing data
        self._t = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        if t is not None:
            self.compute(t, diag=diag)

    def compute(self, t, *, yerr=None, diag=None, check_sorted=True):
        t = tt.as_tensor_variable(t)
        if t.ndim != 1:
            raise ValueError("dimension mismatch")
        if check_sorted:
            t = tt.opt.Assert()(t, tt.all(t[1:] - t[:-1] >= 0))

        # Save the diagonal
        self._t = t
        self._diag = tt.zeros_like(self._t)
        if yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag += tt.as_tensor_variable(yerr) ** 2

        elif diag is not None:
            self._diag += tt.as_tensor_variable(diag)

        # Fill the celerite matrices
        a, self._U, self._V, self._P = self._kernel.get_celerite_matrices(
            self._t, self._diag
        )

        # Compute the Cholesky factorization
        self._d, self._W, _ = ops.factor(a, self._U, self._V, self._P)
        self._log_det = tt.sum(tt.log(self._d))
        self._norm = -0.5 * (
            self._log_det + self._t.shape[0] * np.log(2 * np.pi)
        )

    def _process_input(self, y, *, require_vector=False):
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        y = tt.as_tensor_variable(y)
        if require_vector and y.ndim != 1:
            raise ValueError("'y' must be one dimensional")
        return y

    def apply_inverse(self, y):
        y = self._process_input(y)
        return ops.solve(self._U, self._P, self._d, self._W, y)[0]

    def dot_tril(self, y):
        y = self._process_input(y)
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)[0]

    def log_likelihood(self, y):
        y = self._process_input(y, require_vector=True)
        return (
            self._norm
            - 0.5 * ops.norm(self._U, self._P, self._d, self._W, y)[0]
        )

    def predict(self, y, t=None, *, return_cov=False, return_var=False):
        y = self._process_input(y, require_vector=True)
        alpha = ops.solve(self._U, self._P, self._d, self._W, y)[0]

        if t is None:
            xs = self._t
            mu = y - self._diag * alpha

        else:
            xs = tt.as_tensor_variable(t)
            if xs.ndim != 1:
                raise ValueError("dimension mismatch")

            U_star, V_star, inds = self._kernel.get_conditional_mean_matrices(
                self._t, xs
            )
            mu = ops.conditional_mean(
                self._U, self._V, self._P, alpha, U_star, V_star, inds
            )

        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        KxsT = self._kernel.get_value(xs[None, :] - self._t[:, None])
        if return_var:
            var = self._kernel.get_value(0.0) - tt.batched_dot(
                KxsT.T, self.apply_inverse(KxsT).T
            )
            return mu, var

        # Predictive covariance
        cov = self._kernel.get_value(xs[:, None] - xs[None, :])
        cov -= tt.tensordot(KxsT, self.apply_inverse(KxsT), axes=(0, 0))
        return mu, cov

    def marginal(self, name, **kwargs):
        import pymc3 as pm

        return pm.DensityDist(name, self.log_likelihood, **kwargs)
