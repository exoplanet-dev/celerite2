# -*- coding: utf-8 -*-

__all__ = ["factor", "solve_lower"]

from jax import lax
import jax.numpy as jnp


def factor(a, U, V, P):
    J = U.shape[1]
    Si = jnp.zeros((J, J))
    di = 0.0
    Wi = jnp.zeros(J)
    return lax.scan(_factor_impl, (Si, di, Wi), (a, U, V, P))[1]


def _factor_impl(state, data):
    Sp, dp, Wp = state
    an, Un, Vn, Pn = data
    Sn = Pn[:, None] * (Sp + dp * jnp.outer(Wp, Wp)) * Pn[None, :]
    tmp = Sn @ Un
    dn = an - tmp @ Un
    Wn = (Vn - tmp) / dn
    return (Sn, dn, Wn), (dn, Wn)


def solve_lower(U, W, P, Y):
    J = U.shape[1]
    W0 = jnp.zeros(J)
    if Y.ndim == 1:
        F = jnp.zeros(J)
        Z = 0.0
    else:
        M = Y.shape[1]
        F = jnp.zeros((J, M))
        Z = jnp.zeros(M)
    return lax.scan(_solve_lower_impl, (F, W0, Z), (U, W, P, Y))[1]


def _solve_lower_impl(state, data):
    Fp, Wp, Zp = state
    Un, Wn, Pn, Yn = data
    Fn = Pn[:, None] * (Fp + jnp.outer(Wp, Zp))
    Zn = Yn - Un @ Fn
    return (Fn, Wn, Zn), Zn
