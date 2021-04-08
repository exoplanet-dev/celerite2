# -*- coding: utf-8 -*-

__all__ = [
    "factor",
    "solve_lower",
    "solve_upper",
    "matmul_lower",
    "matmul_upper",
]

import jax.numpy as jnp
from jax import lax


def factor(a, U, V, P):
    J = U.shape[1]
    Si = jnp.zeros((J, J), dtype=a.dtype)
    di = jnp.zeros_like(a[0])
    Wi = jnp.zeros_like(V[0])
    return lax.scan(_factor_impl, (Si, di, Wi), (a, U, V, P))[1]


def solve_lower(U, W, P, Y):
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=W.dtype)
    Wi = jnp.zeros_like(W[0])
    Zi = jnp.zeros_like(Y[0])
    return lax.scan(_solve_impl, (Fi, Wi, Zi), (U, W, P, Y))[1]


def solve_upper(U, W, P, Y):
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=W.dtype)
    Wi = jnp.zeros_like(W[0])
    Zi = jnp.zeros_like(Y[0])
    return lax.scan(
        _solve_impl,
        (Fi, Wi, Zi),
        (W, U, jnp.roll(P, -1, axis=0), Y),
        reverse=True,
    )[1]


def matmul_lower(U, V, P, Y):
    return jnp.einsum("nj,njk->nk", U, _get_matmul_lower_f(V, P, Y))


def matmul_upper(U, V, P, Y):
    return jnp.einsum("nj,njk->nk", V, _get_matmul_upper_f(U, P, Y))


# # General op:
# F = _get_matmul_lower_f(V, P, y)
# inds = np.searchsorted(t, x)
# Pp = jnp.exp(-c[None, :] * (x - t[inds])[:, None])
# Z = jnp.einsum("nj,njk->nk", Pp * Up, F[inds])


#
# Below are the inner update functions that are used in the scan
# implementations of each op above
#


def _pdot(P, other, transpose=False):
    if P.ndim == 1:
        if transpose:
            return other * P[None, :]
        return P[:, None] * other
    if transpose:
        return other @ P
    return P.T @ other


def _factor_impl(state, data):
    Sp, dp, Wp = state
    an, Un, Vn, Pn = data
    Sn = _pdot(Pn, _pdot(Pn, Sp + dp * jnp.outer(Wp, Wp), transpose=True))
    tmp = Sn @ Un
    dn = an - tmp @ Un
    Wn = (Vn - tmp) / dn
    return (Sn, dn, Wn), (dn, Wn)


def _solve_impl(state, data):
    Fp, Wp, Zp = state
    Un, Wn, Pn, Yn = data
    Fn = _pdot(Pn, Fp + jnp.outer(Wp, Zp))
    Zn = Yn - Un @ Fn
    return (Fn, Wn, Zn), Zn


def _matmul_impl(state, data):
    (Fp, Vp, Yp) = state
    Vn, Pn, Yn = data
    Fn = _pdot(Pn, Fp + jnp.outer(Vp, Yp))
    return (Fn, Vn, Yn), Fn


def _get_matmul_lower_f(V, P, Y):
    J = V.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=V.dtype)
    Vi = jnp.zeros_like(V[0])
    Yi = jnp.zeros_like(Y[0])
    return lax.scan(_matmul_impl, (Fi, Vi, Yi), (V, P, Y))[1]


def _get_matmul_upper_f(U, P, Y):
    J = U.shape[1]
    M = Y.shape[1]
    Fi = jnp.zeros((J, M), dtype=U.dtype)
    Ui = jnp.zeros_like(U[0])
    Yi = jnp.zeros_like(Y[0])
    return lax.scan(
        _matmul_impl,
        (Fi, Ui, Yi),
        (U, jnp.roll(P, -1, axis=0), Y),
        reverse=True,
    )[1]
