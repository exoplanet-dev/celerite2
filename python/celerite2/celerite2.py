# -*- coding: utf-8 -*-

__all__ = ["GP"]

import numpy as np

from . import driver


class GP:
    def __init__(self, kernel):
        self._kernel = kernel

        # Placeholders to celerite matrices
        self._U = np.empty((0, 0), dtype=np.float64)
        self._P = np.empty((0, 0), dtype=np.float64)
        self._d = np.empty(0, dtype=np.float64)
        self._W = np.empty((0, 0), dtype=np.float64)

    def compute(self, t, yerr=None, diag=None, check_sorted=True):
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

        elif yerr is None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag[:] = np.atleast_1d(yerr) ** 2

        else:
            self._diag[:] = np.atleast_1d(diag)

        # Allocate memory for the celerite matrices
        N = len(self._t)
        J = self._kernel.J
        self._U.resize((N, J))
        self._P.resize((N - 1, J))
        self._d.resize((N,))
        self._W.resize((N, J))

        # Fill the celerite matrices
        self._kernel.get_celerite_matrices(
            self._t, self._diag, a=self._d, U=self._U, V=self._W, P=self._P
        )

        # Compute the Cholesky factorization
        self._d, self._W = driver.factor(self._U, self._P, self._d, self._W)

    def _process_input(self, y):
        y = np.atleast_1d(y)
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if self._t.shape != y.shape:
            raise ValueError("dimension mismatch")
        return np.ascontiguousarray(y, dtype=np.float64)

    def apply_inverse(self, y, inplace=False):
        y = self._process_input(y)
        if not inplace:
            y = np.copy(y)
        return driver.solve(self._U, self._P, self._d, self._W, y)
