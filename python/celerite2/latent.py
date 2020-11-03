# -*- coding: utf-8 -*-

__all__ = ["apply_latent", "prepare_rectangular_data", "KronLatent"]

import numpy as np


def apply_latent(latent_value, *, a, U, V, P):
    latent_value = np.atleast_3d(latent_value)

    N, J = U.shape
    M = latent_value.shape[2]

    valid_sizes = ((N, J, M), (N, 1, M))
    if latent_value.shape not in valid_sizes:
        raise ValueError(
            "The 'latent' function returned an invalid shape; "
            f"expected ({N}, {J}, {M}) or expected ({N}, 1, {M}), "
            f"got {latent_value.shape}"
        )

    a[:] *= np.sum(latent_value ** 2, axis=(1, 2))
    if M == 1:
        U[:] *= latent_value[:, :, 0]
        V[:] *= latent_value[:, :, 0]
    else:
        U = (U[:, :, None] * latent_value).reshape((N, -1))
        V = (V[:, :, None] * latent_value).reshape((N, -1))
        P = np.tile(P[:, :, None], (1, 1, M)).reshape(N - 1, -1)

    return a, U, V, P


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


class KronLatent:
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
