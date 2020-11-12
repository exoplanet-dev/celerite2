# -*- coding: utf-8 -*-

__all__ = ["LatentTerm", "prepare_rectangular_data", "KroneckerLatent"]
import numpy as np

from .terms import Term


class LatentTerm(Term):
    __requires_general_addition__ = True

    def __init__(self, term, left=None, right=None):
        self.term = term
        self.left = left
        self.right = right

    def _get_latent(self, left_or_right, t, X):
        if left_or_right is None:
            return np.ones((1, 1, 1))
        if callable(left_or_right):
            left_or_right = left_or_right(t, X)
        else:
            left_or_right = np.atleast_1d(left_or_right)
        if left_or_right.ndim == 1:
            return left_or_right[None, None, :]
        if left_or_right.ndim == 2:
            return left_or_right[None, :, :]
        if left_or_right.ndim != 3:
            raise ValueError("Invalid shape for latent")
        return left_or_right

    def get_left_latent(self, t, X):
        return self._get_latent(self.left, t, X)

    def get_right_latent(self, t, X):
        if self.right is None and self.left is not None:
            return self._get_latent(self.left, t, X)
        return self._get_latent(self.right, t, X)

    def get_celerite_matrices(
        self,
        t,
        diag,
        *,
        c=None,
        a=None,
        U=None,
        V=None,
        X=None,
    ):
        t = np.atleast_1d(t)
        diag = np.atleast_1d(diag)
        c0, a0, U0, V0 = self.term.get_celerite_matrices(t, diag, X=X)

        left = self.get_left_latent(t, X)
        right = self.get_right_latent(t, X)

        N = len(t)
        K = left.shape[2]
        J = c0.shape[0] * K
        c, a, U, V = self._resize_matrices(N, J, c, a, U, V)

        c[:] = np.tile(c0, K)
        U[:] = np.ascontiguousarray(U0[:, :, None] * left).reshape((N, -1))
        V[:] = np.ascontiguousarray(V0[:, :, None] * right).reshape((N, -1))
        a[:] = diag + np.sum(U * V, axis=-1)

        return c, a, U, V


def prepare_rectangular_data(N, M, t, **kwargs):
    results = dict(
        t=np.tile(np.asarray(t)[:, None], (1, M)).flatten(),
        X=np.tile(np.arange(M), N),
    )

    for k, v in kwargs.items():
        results[k] = np.ascontiguousarray(
            np.broadcast_to(v, (N, M)), dtype=np.float64
        ).flatten()

    return results


class KroneckerLatent:
    def __init__(self, *, R=None, L=None):
        if R is not None:
            if L is not None:
                raise ValueError("Only one of 'R' and 'L' can be defined")
            R = np.ascontiguousarray(np.atleast_2d(R), dtype=np.float64)
            try:
                self.L = np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                M = np.copy(R)
                M[np.diag_indices_from(M)] += 1e-10
                self.L = np.linalg.cholesky(M)
        elif L is not None:
            self.L = np.ascontiguousarray(L, dtype=np.float64)
            if self.L.ndim == 1:
                self.L = np.reshape(self.L, (-1, 1))
        else:
            raise ValueError("One of 'R' and 'L' must be defined")

    def __call__(self, t, inds):
        return self.L[inds][:, None, :]
