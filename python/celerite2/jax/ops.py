# -*- coding: utf-8 -*-

"""
Note: for better performance, these should be implemented as JAX primitives
with custom XLA calls:

- https://jax.readthedocs.io/en/stable/notebooks/How_JAX_primitives_work.html
- https://github.com/danieljtait/jax_xla_adventures
"""

__all__ = [
    "factor",
    "solve",
    "norm",
    "dot_tril",
    "matmul",
    "conditional_mean",
]
import logging
from functools import wraps

import numpy as np

from .. import driver, ext

logger = logging.getLogger(__name__)

from jax.config import config  # noqa isort:skip

if not config.read("jax_enable_x64"):
    logger.warning(
        "celerite2.jax only works with dtype float64. "
        "To enable, run (before importing jax or celerite2.jax):\n"
        ">>> from jax.config import config\n"
        ">>> config.update('jax_enable_x64', True)"
    )

import jax  # noqa isort:skip
import jax.numpy as jnp  # noqa isort:skip


def to_jax(x):
    return jnp.asarray(x)


def to_np(x):
    return np.asarray(x)


class wrap_impl:
    def __init__(self, num_out=None):
        self.num_out = num_out

    def __call__(self, func):
        if self.num_out == 1:

            @wraps(func)
            def wrapped(*args):
                return to_jax(func(*map(to_np, args)))

        else:

            @wraps(func)
            def wrapped(*args):
                out = func(*map(to_np, args))
                return tuple(map(to_jax, out))

        return wrapped


class wrap_fwd:
    def __init__(self, num_out):
        self.num_out = int(num_out)

    def __call__(self, func):
        if self.num_out == 1:

            @wraps(func)
            def wrapped(*args):
                in_arrays = tuple(map(to_np, args))
                out_arrays = func(*in_arrays)
                out_tensors = tuple(map(to_jax, out_arrays[: self.num_out]))
                return (
                    out_tensors[0],
                    in_arrays + out_arrays,
                )

        else:

            @wraps(func)
            def wrapped(*args):
                in_arrays = tuple(map(to_np, args))
                out_arrays = func(*in_arrays)
                out_tensors = tuple(map(to_jax, out_arrays[: self.num_out]))
                return (
                    out_tensors,
                    in_arrays + out_arrays,
                )

        return wrapped


class wrap_rev:
    def __init__(self, num_out):
        self.num_out = int(num_out)

    def __call__(self, func):
        if self.num_out == 1:

            @wraps(func)
            def wrapped(args, grads):
                return tuple(map(to_jax, func(*args, to_np(grads))))

        else:

            @wraps(func)
            def wrapped(args, grads):
                return tuple(map(to_jax, func(*args, *map(to_np, grads))))

        return wrapped


@jax.custom_vjp
@wrap_impl(2)
def factor(a, U, V, P):
    return driver.factor(U, P, np.copy(a), np.copy(V))


factor.defvjp(wrap_fwd(2)(ext.factor_fwd), wrap_rev(2)(ext.factor_rev))


@jax.custom_vjp
@wrap_impl(1)
def solve(U, P, d, W, Y):
    return driver.solve(U, P, d, W, np.copy(Y))


solve.defvjp(wrap_fwd(1)(ext.solve_fwd), wrap_rev(1)(ext.solve_rev))


@jax.custom_vjp
@wrap_impl(1)
def norm(U, P, d, W, Y):
    return driver.norm(U, P, d, W, np.copy(Y))


norm.defvjp(wrap_fwd(1)(ext.norm_fwd), wrap_rev(1)(ext.norm_rev))


@jax.custom_vjp
@wrap_impl(1)
def dot_tril(U, P, d, W, Y):
    return driver.dot_tril(U, P, d, W, np.copy(Y))


dot_tril.defvjp(wrap_fwd(1)(ext.dot_tril_fwd), wrap_rev(1)(ext.dot_tril_rev))


@jax.custom_vjp
@wrap_impl(1)
def matmul(a, U, V, P, Y):
    return driver.matmul(a, U, V, P, Y, np.empty_like(Y))


matmul.defvjp(wrap_fwd(1)(ext.matmul_fwd), wrap_rev(1)(ext.matmul_rev))


@wrap_impl(1)
def conditional_mean(U, V, P, z, U_star, V_star, inds):
    mu = np.empty(len(inds), dtype=np.float64)
    return driver.conditional_mean(U, V, P, z, U_star, V_star, inds, mu)
