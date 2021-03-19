# -*- coding: utf-8 -*-

__all__ = ["factor", "solve_lower"]

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


def _factor_impl(state, data):
    Sp, dp, Wp = state
    an, Un, Vn, Pn = data
    Sn = Pn[:, None] * (Sp + dp * jnp.outer(Wp, Wp)) * Pn[None, :]
    tmp = Sn @ Un
    dn = an - tmp @ Un
    Wn = (Vn - tmp) / dn
    return (Sn, dn, Wn), (dn, Wn)


def _solve_impl(state, data):
    Fp, Wp, Zp = state
    Un, Wn, Pn, Yn = data
    Fn = Pn[:, None] * (Fp + jnp.outer(Wp, Zp))
    Zn = Yn - Un @ Fn
    return (Fn, Wn, Zn), Zn
