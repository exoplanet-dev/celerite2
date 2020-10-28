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
from functools import wraps

import jax
import numpy as np
from jax import core, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, xla
from jax.lib import xla_client

from .. import driver, ext
from . import xla_ops

xops = xla_client.ops

xla_client.register_cpu_custom_call_target(
    b"celerite2_factor", xla_ops.factor()
)
xla_client.register_cpu_custom_call_target(
    b"celerite2_factor_rev", xla_ops.factor_rev()
)


def factor(a, U, V, P):
    flag, d, W, S = factor_p.bind(a, U, V, P)
    return d, W


def factor_impl(a, U, V, P):
    return xla.apply_primitive(factor_p, a, U, V, P)


def factor_abstract_eval(a, U, V, P):
    N, J = U.shape
    if a.shape != (N,):
        raise ValueError(
            f"Invalid shape for 'a'; expected {(N,)}, got {a.shape}"
        )
    if V.shape != (N, J):
        raise ValueError(
            f"Invalid shape for 'V'; expected {(N, J)}, got {V.shape}"
        )
    if P.shape != (N - 1, J):
        raise ValueError(
            f"Invalid shape for 'P'; expected {(N - 1, J)}, got {P.shape}"
        )
    if (
        a.dtype != jnp.float64
        or U.dtype != jnp.float64
        or V.dtype != jnp.float64
        or P.dtype != jnp.float64
    ):
        raise ValueError("Invalid dtype; must be float64")
    return [
        ShapedArray((), jnp.int32),
        ShapedArray(a.shape, jnp.float64),
        ShapedArray(V.shape, jnp.float64),
        ShapedArray((N, J, J), jnp.float64),
    ]


def factor_translation_rule(c, a, U, V, P):
    shape = c.get_shape(U)

    N, J = shape.dimensions()
    dtype = shape.element_type()
    if dtype != np.float64:
        raise ValueError("Invalid dtype; must be float64")

    return xops.CustomCallWithLayout(
        c,
        b"celerite2_factor",
        operands=(
            xops.ConstantLiteral(c, np.int32(N)),
            xops.ConstantLiteral(c, np.int32(J)),
            a,
            U,
            V,
            P,
        ),
        shape_with_layout=xla_client.Shape.tuple_shape(
            (
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.int32),
                    (),
                    (),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N,),
                    (0,),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N, J),
                    (1, 0),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N, J, J),
                    (
                        2,
                        1,
                        0,
                    ),
                ),
            )
        ),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.float64), (N,), (0,)),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N - 1, J), (1, 0)
            ),
        ),
    )


def factor_and_jvp(arg_values, arg_tangents):
    a, U, V, P = arg_values
    at, Ut, Vt, Pt = arg_tangents
    flag, d, W, S = factor_p.bind(a, U, V, P)

    def make_zero(x, tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan

    at, Ut, Vt, Pt = (
        make_zero(a, at),
        make_zero(U, Ut),
        make_zero(V, Vt),
        make_zero(P, Pt),
    )
    dt, Wt = factor_jvp_p.bind(a, U, V, P, d, W, S, at, Ut, Vt, Pt)
    return ((flag, d, W, S), (None, dt, Wt, None))


def factor_jvp_abstract_eval(a, U, V, P, d, W, S, at, Ut, Vt, Pt):
    return [
        ShapedArray(d.shape, jnp.float64),
        ShapedArray(W.shape, jnp.float64),
    ]


def factor_jvp_transpose(ct, a, U, V, P, d, W, S, at, Ut, Vt, Pt):
    def make_zero(x, tan):
        return lax.zeros_like_array(x) if type(tan) is ad.Zero else tan

    bd, bW = ct
    bd = make_zero(d, bd)
    bW = make_zero(W, bW)
    ba, bU, bV, bP = factor_rev_p.bind(a, U, V, P, d, W, S, bd, bW)
    return None, None, None, None, None, None, None, ba, bU, bV, bP


def factor_rev_impl(a, U, V, P, d, W, S, bd, bW):
    a, U, V, P, d, W, S, bd, bW = map(to_np, (a, U, V, P, d, W, S, bd, bW))
    S = np.reshape(S, (len(S), -1))
    ba, bU, bV, bW = ext.factor_rev(a, U, V, P, d, W, S, bd, bW)
    return tuple(map(to_jax, (ba, bU, bV, bW)))


def factor_rev_abstract_eval(a, U, V, P, d, W, S, bd, bW):
    return [
        ShapedArray(a.shape, jnp.float64),
        ShapedArray(U.shape, jnp.float64),
        ShapedArray(V.shape, jnp.float64),
        ShapedArray(P.shape, jnp.float64),
    ]


def factor_rev_translation_rule(c, a, U, V, P, d, W, S, bd, bW):
    shape = c.get_shape(U)
    N, J = shape.dimensions()
    dtype = shape.element_type()
    if dtype != np.float64:
        raise ValueError("Invalid dtype; must be float64")
    return xops.CustomCallWithLayout(
        c,
        b"celerite2_factor_rev",
        operands=(
            xops.ConstantLiteral(c, np.int32(N)),
            xops.ConstantLiteral(c, np.int32(J)),
            a,
            U,
            V,
            P,
            d,
            W,
            S,
            bd,
            bW,
        ),
        shape_with_layout=xla_client.Shape.tuple_shape(
            (
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N,),
                    (0,),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N, J),
                    (1, 0),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N, J),
                    (1, 0),
                ),
                xla_client.Shape.array_shape(
                    jnp.dtype(jnp.float64),
                    (N - 1, J),
                    (
                        1,
                        0,
                    ),
                ),
            )
        ),
        operand_shapes_with_layout=(
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ()),
            xla_client.Shape.array_shape(jnp.dtype(jnp.float64), (N,), (0,)),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N - 1, J), (1, 0)
            ),
            xla_client.Shape.array_shape(jnp.dtype(jnp.float64), (N,), (0,)),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J, J), (2, 1, 0)
            ),
            xla_client.Shape.array_shape(jnp.dtype(jnp.float64), (N,), (0,)),
            xla_client.Shape.array_shape(
                jnp.dtype(jnp.float64), (N, J), (1, 0)
            ),
        ),
    )


factor_p = core.Primitive("celerite2_factor")
factor_p.multiple_results = True
factor_p.def_impl(factor_impl)
factor_p.def_abstract_eval(factor_abstract_eval)
xla.backend_specific_translations["cpu"][factor_p] = factor_translation_rule
ad.primitive_jvps[factor_p] = factor_and_jvp

factor_jvp_p = core.Primitive("celerite2_factor_jvp")
factor_jvp_p.multiple_results = True
factor_jvp_p.def_abstract_eval(factor_jvp_abstract_eval)
ad.primitive_transposes[factor_jvp_p] = factor_jvp_transpose

factor_rev_p = core.Primitive("celerite2_factor_rev")
factor_rev_p.multiple_results = True
factor_rev_p.def_impl(factor_rev_impl)
factor_rev_p.def_abstract_eval(factor_rev_abstract_eval)
xla.backend_specific_translations["cpu"][
    factor_rev_p
] = factor_rev_translation_rule


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


# @jax.custom_vjp
# @wrap_impl(2)
# def factor(a, U, V, P):
#     return driver.factor(U, P, np.copy(a), np.copy(V))


# factor.defvjp(wrap_fwd(2)(ext.factor_fwd), wrap_rev(2)(ext.factor_rev))


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
