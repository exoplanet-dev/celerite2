# -*- coding: utf-8 -*-

__all__ = ["get_matrices"]
import numpy as np

from . import terms


def get_matrices(
    size=100, kernel=None, vector=False, conditional=False, include_dense=False
):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    if vector:
        Y = np.sin(x)
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
        )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    if include_dense:
        K = kernel.get_value(x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += diag

    if not conditional:
        if include_dense:
            return a, U, V, P, K, Y
        return a, U, V, P, Y

    t = np.sort(np.random.uniform(-1, 12, 200))
    U_star, V_star, inds = kernel.get_conditional_mean_matrices(x, t)

    if include_dense:
        K_star = kernel.get_value(t[:, None] - x[None, :])
        return a, U, V, P, K, Y, U_star, V_star, inds, K_star

    return a, U, V, P, Y, U_star, V_star, inds
