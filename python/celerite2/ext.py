# -*- coding: utf-8 -*-

__all__ = [
    "factor_fwd",
    "factor_rev",
    "solve_fwd",
    "solve_rev",
    "norm_fwd",
    "norm_rev",
    "dot_tril_fwd",
    "dot_tril_rev",
    "matmul_fwd",
    "matmul_rev",
    "BaseGaussianProcess",
]
import numpy as np

from . import backprop
from .celerite2 import GaussianProcess


def factor_fwd(a, U, V, P):
    N, J = U.shape
    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((N, J ** 2), dtype=np.float64)
    return backprop.factor_fwd(a, U, V, P, d, W, S)


def factor_rev(a, U, V, P, d, W, S, bd, bW):
    ba = np.empty_like(a)
    bU = np.empty_like(U)
    bV = np.empty_like(V)
    bP = np.empty_like(P)
    return backprop.factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP)


def solve_fwd(U, P, d, W, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    Z = np.empty_like(X)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    G = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.solve_fwd(U, P, d, W, Y, X, Z, F, G)


def solve_rev(U, P, d, W, Y, X, Z, F, G, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.solve_rev(
        U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY
    )


def norm_fwd(U, P, d, W, Y):
    N, J = U.shape
    X = np.empty((), dtype=np.float64)
    Z = np.empty_like(Y)
    F = np.empty((N, J), dtype=np.float64)
    return backprop.norm_fwd(U, P, d, W, Y, X, Z, F)


def norm_rev(U, P, d, W, Y, X, Z, F, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.norm_rev(U, P, d, W, Y, X, Z, F, bX, bU, bP, bd, bW, bY)


def dot_tril_fwd(U, P, d, W, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.dot_tril_fwd(U, P, d, W, Y, X, F)


def dot_tril_rev(U, P, d, W, Y, X, F, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.dot_tril_rev(U, P, d, W, Y, X, F, bX, bU, bP, bd, bW, bY)


def matmul_fwd(a, U, V, P, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    Z = np.empty_like(X)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    G = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.matmul_fwd(a, U, V, P, Y, X, Z, F, G)


def matmul_rev(a, U, V, P, Y, X, Z, F, G, bX):
    ba = np.empty_like(a)
    bU = np.empty_like(U)
    bV = np.empty_like(V)
    bP = np.empty_like(P)
    bY = np.empty_like(Y)
    return backprop.matmul_rev(
        a, U, V, P, Y, X, Z, F, G, bX, ba, bU, bV, bP, bY
    )


class BaseGaussianProcess(GaussianProcess):
    def __init__(self, kernel, t=None, *, mean=0.0, **kwargs):
        self.kernel = kernel
        self.mean = mean

        # Placeholders for storing data
        self._t = None
        self._mean_value = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        if t is not None:
            self.compute(t, **kwargs)

    def as_tensor(self, tensor):
        raise NotImplementedError("must be implemented by subclasses")

    def zeros_like(self, tensor):
        raise NotImplementedError("must be implemented by subclasses")

    def do_compute(self, quiet):
        raise NotImplementedError("must be implemented by subclasses")

    def check_sorted(self, t):
        raise NotImplementedError("must be implemented by subclasses")

    def do_solve(self, y):
        raise NotImplementedError("must be implemented by subclasses")

    def do_dot_tril(self, y):
        raise NotImplementedError("must be implemented by subclasses")

    def do_norm(self, y):
        raise NotImplementedError("must be implemented by subclasses")

    def do_conditional_mean(self, *args):
        raise NotImplementedError("must be implemented by subclasses")

    def tensordot(self, a, b):
        raise NotImplementedError("must be implemented by subclasses")

    def diagdot(self, a, b):
        raise NotImplementedError("must be implemented by subclasses")

    def compute(
        self, t, *, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        t = self.as_tensor(t)
        if t.ndim != 1:
            raise ValueError("dimension mismatch")
        if check_sorted:
            t = self.check_sorted(t)

        # Save the diagonal
        self._t = self.as_tensor(t)
        self._mean_value = self._mean(self._t)
        self._diag = self.zeros_like(self._t)
        if yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag += self.as_tensor(yerr) ** 2

        elif diag is not None:
            self._diag += self.as_tensor(diag)

        # Fill the celerite matrices
        (
            self._a,
            self._U,
            self._V,
            self._P,
        ) = self.kernel.get_celerite_matrices(self._t, self._diag)

        self.do_compute(quiet)

    def _process_input(self, y, *, require_vector=False):
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        y = self.as_tensor(y)
        if require_vector and y.ndim != 1:
            raise ValueError("'y' must be one dimensional")
        return y

    def apply_inverse(self, y):
        y = self._process_input(y)
        return self.do_solve(y)

    def dot_tril(self, y):
        y = self._process_input(y)
        return self.do_dot_tril(y)

    def log_likelihood(self, y):
        y = self._process_input(y, require_vector=True)
        return self._norm - 0.5 * self.do_norm(y - self._mean_value)

    def predict(
        self,
        y,
        t=None,
        *,
        return_cov=False,
        return_var=False,
        include_mean=True,
        kernel=None,
        _fast_mean=True,
    ):
        y = self._process_input(y, require_vector=True)
        resid = y - self._mean_value
        alpha = self.do_solve(resid)

        if t is None:
            xs = self._t

        else:
            xs = self.as_tensor(t)
            if xs.ndim != 1:
                raise ValueError("dimension mismatch")

        KxsT = None
        mu = None
        if kernel is None:
            kernel = self.kernel

            if t is None:
                if include_mean:
                    mu = y - self._diag * alpha
                else:
                    mu = resid - self._diag * alpha

            elif _fast_mean:
                (
                    U_star,
                    V_star,
                    inds,
                ) = self.kernel.get_conditional_mean_matrices(self._t, xs)
                mu = self.do_conditional_mean(alpha, U_star, V_star, inds)

                if include_mean:
                    mu += self._mean(xs)

        if mu is None:
            if kernel is None:
                kernel = self.kernel
            KxsT = kernel.get_value(xs[None, :] - self._t[:, None])
            mu = self.tensordot(KxsT, alpha)
            if include_mean:
                mu += self._mean(xs)

        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        if KxsT is None:
            KxsT = kernel.get_value(xs[None, :] - self._t[:, None])
        if return_var:
            var = kernel.get_value(0.0) - self.diagdot(
                KxsT, self.do_solve(KxsT)
            )
            return mu, var

        # Predictive covariance
        cov = kernel.get_value(xs[:, None] - xs[None, :])
        cov -= self.tensordot(KxsT, self.do_solve(KxsT))
        return mu, cov

    def sample(self, *args, **kwargs):
        raise NotImplementedError("'sample' is not implemented by extensions")

    def sample_conditional(self, *args, **kwargs):
        raise NotImplementedError(
            "'sample_conditional' is not implemented by extensions"
        )
