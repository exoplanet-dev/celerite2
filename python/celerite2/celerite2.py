# -*- coding: utf-8 -*-

__all__ = ["GP"]

import numpy as np

from . import driver
from .driver import LinAlgError


class GP:
    def __init__(self, kernel):
        self._kernel = kernel

        # Placeholders for storing data
        self._t = None
        self._diag = None

        # Placeholders to celerite matrices
        self._U = np.empty((0, 0), dtype=np.float64)
        self._P = np.empty((0, 0), dtype=np.float64)
        self._d = np.empty(0, dtype=np.float64)
        self._W = np.empty((0, 0), dtype=np.float64)

    def compute(self, t, yerr=None, diag=None, check_sorted=True, quiet=False):
        # Check the input coordinates
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if len(t.shape) != 1:
            raise ValueError("dimension mismatch")

        # Save the diagonal
        self._t = np.ascontiguousarray(t, dtype=np.float64)
        self._diag = np.empty_like(self._t)
        if yerr is None and diag is None:
            self._diag[:] = 0.0

        elif yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag[:] = np.atleast_1d(yerr) ** 2

        else:
            self._diag[:] = np.atleast_1d(diag)

        # Fill the celerite matrices
        (
            self._d,
            self._U,
            self._W,
            self._P,
        ) = self._kernel.get_celerite_matrices(
            self._t, self._diag, a=self._d, U=self._U, V=self._W, P=self._P
        )

        # Compute the Cholesky factorization
        try:
            self._d, self._W = driver.factor(
                self._U, self._P, self._d, self._W
            )
        except LinAlgError:
            if not quiet:
                raise
            self._log_det = -np.inf
            self._norm = np.inf
        else:
            self._log_det = np.sum(np.log(self._d))
            self._norm = -0.5 * (
                self._log_det + len(self._t) * np.log(2 * np.pi)
            )

    def recompute(self, quiet=False):
        if self._t is None:
            raise RuntimeError(
                "you must call 'compute' directly  at least once"
            )
        return self.compute(
            self._t, diag=self._diag, check_sorted=False, quiet=quiet
        )

    def _process_input(self, y, inplace=False):
        y = np.atleast_1d(y)
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if self._t.shape[0] != y.shape[0]:
            raise ValueError("dimension mismatch")
        y = np.ascontiguousarray(y, dtype=np.float64)
        if not inplace:
            y = np.copy(y)
        return y

    def apply_inverse(self, y, inplace=False):
        y = self._process_input(y, inplace=inplace)
        return driver.solve(self._U, self._P, self._d, self._W, y)

    def log_likelihood(self, y, inplace=False):
        y = self._process_input(y, inplace=inplace)
        if self._t.shape != y.shape:
            raise ValueError("'y' must be one dimensional")
        if not np.isfinite(self._log_det):
            return -np.inf
        loglike = self._norm - 0.5 * driver.norm(
            self._U, self._P, self._d, self._W, y
        )
        if not np.isfinite(loglike):
            return -np.inf
        return loglike
