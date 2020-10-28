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
from collections import OrderedDict
from functools import partial

import numpy as np
from jax import core, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, xla
from jax.lib import xla_client

from . import xla_ops

xops = xla_client.ops


def factor(a, U, V, P):
    d, W, S = factor_prim.bind(a, U, V, P)
    return d, W


def solve(U, P, d, W, Y):
    if Y.ndim == 1:
        X, Z, F, G = solve_vector_prim.bind(U, P, d, W, Y)
    else:
        X, Z, F, G = solve_prim.bind(U, P, d, W, Y)
    return X


def norm(U, P, d, W, Y):
    X, Z, F = norm_prim.bind(U, P, d, W, Y)
    return X


def dot_tril(U, P, d, W, Y):
    if Y.ndim == 1:
        X, F = dot_tril_vector_prim.bind(U, P, d, W, Y)
    else:
        X, F = dot_tril_prim.bind(U, P, d, W, Y)
    return X


def matmul(a, U, V, P, Y):
    if Y.ndim == 1:
        X, Z, F, G = matmul_vector_prim.bind(a, U, V, P, Y)
    else:
        X, Z, F, G = matmul_prim.bind(a, U, V, P, Y)
    return X


def conditional_mean(U, V, P, z, U_star, V_star, inds):
    (mu,) = conditional_mean_prim.bind(U, V, P, z, U_star, V_star, inds)
    return mu


def _abstract_eval(spec, *args):
    vals = spec["get_dims"](*(a.shape for a in args))
    for s, arg in zip(spec["inputs"], args):
        if arg.dtype != s.get("dtype", np.float64):
            raise ValueError(
                f"Invalid dtype for '{s['name']}'; "
                f"expected {s.get('dtype', np.float64)}, got {arg.dtype}"
            )
        shape = eval(s["shape"], dict(vals))
        if arg.shape != shape:
            raise ValueError(
                f"Invalid shape for '{s['name']}'; "
                f"expected {shape}, got {arg.shape}"
            )
    return tuple(
        ShapedArray(eval(s["shape"], dict(vals)), s.get("dtype", np.float64))
        for s in spec["outputs"] + spec["extra_outputs"]
    )


def _translation_rule(spec, c, *args):
    vals = spec["get_dims"](*(c.get_shape(a).dimensions() for a in args))
    dtype = c.get_shape(args[0]).element_type()
    if dtype != np.float64:
        raise ValueError("Invalid dtype; must be float64")

    return xops.CustomCallWithLayout(
        c,
        spec["xla_name"],
        operands=tuple(
            xops.ConstantLiteral(c, np.int32(v)) for v in vals.values()
        )
        + args,
        shape_with_layout=xla_client.Shape.tuple_shape(
            tuple(
                xla_client.Shape.array_shape(
                    jnp.dtype(dtype),
                    shape,
                    tuple(range(len(shape) - 1, -1, -1)),
                )
                for dtype, shape in (
                    (
                        s.get("dtype", np.float64),
                        eval(s["shape"], dict(vals)),
                    )
                    for s in spec["outputs"] + spec["extra_outputs"]
                )
            )
        ),
        operand_shapes_with_layout=tuple(
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ())
            for _ in range(len(vals))
        )
        + tuple(
            xla_client.Shape.array_shape(
                jnp.dtype(dtype),
                shape,
                tuple(range(len(shape) - 1, -1, -1)),
            )
            for dtype, shape in (
                (s.get("dtype", np.float64), eval(s["shape"], dict(vals)))
                for s in spec["inputs"]
            )
        ),
    )


def _jvp(spec, arg_values, arg_tangents):
    def make_zero(x, t):
        return lax.zeros_like_array(x) if type(t) is ad.Zero else t

    out_values = tuple(spec["base_primitive"].bind(*arg_values))
    arg_tangents = tuple(
        make_zero(x, t) for x, t in zip(arg_values, arg_tangents)
    )
    out_tangents = tuple(
        spec["jvp_primitive"].bind(*(arg_values + out_values + arg_tangents))
    )
    return out_values, out_tangents + tuple(
        None for _ in spec["extra_outputs"]
    )


def _jvp_abstract_eval(spec, *args):
    return tuple(
        ShapedArray(arg.shape, arg.dtype)
        for arg in args[
            len(spec["inputs"]) : len(spec["inputs"]) + len(spec["outputs"])
        ]
    )


def _jvp_transpose(spec, c_out, *args):
    def make_zero(x, t):
        return lax.zeros_like_array(x) if type(t) is ad.Zero else t

    nin = len(spec["inputs"])
    nout = len(spec["outputs"])
    nargs = nin + nout + len(spec["extra_outputs"])
    c_out = tuple(
        make_zero(x, t) for x, t in zip(args[nin : nin + nout], c_out)
    )
    c_in = tuple(spec["rev_primitive"].bind(*(args[:nargs] + c_out)))
    return tuple(None for _ in range(nargs)) + c_in


def _rev_abstract_eval(spec, *args):
    return tuple(
        ShapedArray(arg.shape, arg.dtype)
        for arg in args[: len(spec["inputs"])]
    )


def _rev_translation_rule(spec, c, *args):
    rev_spec = dict(
        xla_name=spec["xla_name"] + b"_rev",
        get_dims=spec["get_dims"],
        inputs=spec["inputs"]
        + spec["outputs"]
        + spec["extra_outputs"]
        + spec["outputs"],
        outputs=spec["inputs"],
        extra_outputs=(),
    )
    return _translation_rule(rev_spec, c, *args)


def setup_spec(spec, grad=True):
    xla_client.register_cpu_custom_call_target(
        spec["xla_name"], getattr(xla_ops, spec["name"])()
    )

    prim = core.Primitive("celerite2_" + spec["name"])
    prim.multiple_results = True
    spec["base_primitive"] = prim

    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(partial(_abstract_eval, spec))
    xla.backend_specific_translations["cpu"][prim] = partial(
        _translation_rule, spec
    )

    if not grad:
        return prim

    xla_client.register_cpu_custom_call_target(
        spec["xla_name"] + b"_rev", getattr(xla_ops, spec["name"] + "_rev")()
    )

    jvp = core.Primitive("celerite2_" + spec["name"] + "_jvp")
    jvp.multiple_results = True
    rev = core.Primitive("celerite2_" + spec["name"] + "_rev")
    rev.multiple_results = True
    spec["jvp_primitive"] = jvp
    spec["rev_primitive"] = rev

    ad.primitive_jvps[prim] = partial(_jvp, spec)
    jvp.def_abstract_eval(partial(_jvp_abstract_eval, spec))
    ad.primitive_transposes[jvp] = partial(_jvp_transpose, spec)

    rev.def_impl(partial(xla.apply_primitive, rev))
    rev.def_abstract_eval(partial(_rev_abstract_eval, spec))
    xla.backend_specific_translations["cpu"][rev] = partial(
        _rev_translation_rule, spec
    )

    return prim


factor_prim = setup_spec(
    dict(
        name="factor",
        xla_name=b"celerite2_factor",
        get_dims=lambda *args: OrderedDict(list(zip(("N", "J"), args[1]))),
        inputs=(
            dict(name="a", shape="(N,)"),
            dict(name="U", shape="(N, J)"),
            dict(name="V", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
        ),
        outputs=(
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
        ),
        extra_outputs=(dict(name="S", shape="(N, J, J)"),),
    )
)

solve_prim = setup_spec(
    dict(
        name="solve",
        xla_name=b"celerite2_solve",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[0])) + [("nrhs", args[4][1])]
        ),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
            dict(name="Y", shape="(N, nrhs)"),
        ),
        outputs=(dict(name="X", shape="(N, nrhs)"),),
        extra_outputs=(
            dict(name="Z", shape="(N, nrhs)"),
            dict(name="F", shape="(N, J, nrhs)"),
            dict(name="G", shape="(N, J, nrhs)"),
        ),
    )
)
solve_vector_prim = setup_spec(
    dict(
        name="solve",
        xla_name=b"celerite2_solve",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[0])) + [("nrhs", 1)]
        ),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
            dict(name="Y", shape="(N,)"),
        ),
        outputs=(dict(name="X", shape="(N,)"),),
        extra_outputs=(
            dict(name="Z", shape="(N,)"),
            dict(name="F", shape="(N, J)"),
            dict(name="G", shape="(N, J)"),
        ),
    )
)

norm_prim = setup_spec(
    dict(
        name="norm",
        xla_name=b"celerite2_norm",
        get_dims=lambda *args: OrderedDict(list(zip(("N", "J"), args[0]))),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
            dict(name="Y", shape="(N,)"),
        ),
        outputs=(dict(name="X", shape="()"),),
        extra_outputs=(
            dict(name="Z", shape="(N,)"),
            dict(name="F", shape="(N, J)"),
        ),
    )
)

dot_tril_prim = setup_spec(
    dict(
        name="dot_tril",
        xla_name=b"celerite2_dot_tril",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[0])) + [("nrhs", args[4][1])]
        ),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
            dict(name="Y", shape="(N, nrhs)"),
        ),
        outputs=(dict(name="X", shape="(N, nrhs)"),),
        extra_outputs=(dict(name="F", shape="(N, J, nrhs)"),),
    )
)
dot_tril_vector_prim = setup_spec(
    dict(
        name="dot_tril",
        xla_name=b"celerite2_dot_tril",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[0])) + [("nrhs", 1)]
        ),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="d", shape="(N,)"),
            dict(name="W", shape="(N, J)"),
            dict(name="Y", shape="(N,)"),
        ),
        outputs=(dict(name="X", shape="(N,)"),),
        extra_outputs=(dict(name="F", shape="(N, J)"),),
    )
)

matmul_prim = setup_spec(
    dict(
        name="matmul",
        xla_name=b"celerite2_matmul",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[1])) + [("nrhs", args[4][1])]
        ),
        inputs=(
            dict(name="a", shape="(N,)"),
            dict(name="U", shape="(N, J)"),
            dict(name="V", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="Y", shape="(N, nrhs)"),
        ),
        outputs=(dict(name="X", shape="(N, nrhs)"),),
        extra_outputs=(
            dict(name="Z", shape="(N, nrhs)"),
            dict(name="F", shape="(N, J, nrhs)"),
            dict(name="G", shape="(N, J, nrhs)"),
        ),
    )
)
matmul_vector_prim = setup_spec(
    dict(
        name="matmul",
        xla_name=b"celerite2_matmul",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[1])) + [("nrhs", 1)]
        ),
        inputs=(
            dict(name="a", shape="(N,)"),
            dict(name="U", shape="(N, J)"),
            dict(name="V", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="Y", shape="(N,)"),
        ),
        outputs=(dict(name="X", shape="(N,)"),),
        extra_outputs=(
            dict(name="Z", shape="(N,)"),
            dict(name="F", shape="(N, J)"),
            dict(name="G", shape="(N, J)"),
        ),
    )
)

conditional_mean_prim = setup_spec(
    dict(
        name="conditional_mean",
        xla_name=b"celerite2_conditional_mean",
        get_dims=lambda *args: OrderedDict(
            list(zip(("N", "J"), args[1])) + [("M", args[6][0])]
        ),
        inputs=(
            dict(name="U", shape="(N, J)"),
            dict(name="V", shape="(N, J)"),
            dict(name="P", shape="(N - 1, J)"),
            dict(name="Z", shape="(N,)"),
            dict(name="U_star", shape="(M, J)"),
            dict(name="V_star", shape="(M, J)"),
            dict(name="inds", shape="(M,)", dtype=np.int64),
        ),
        outputs=(dict(name="mu", shape="(M,)"),),
        extra_outputs=(),
    ),
    grad=False,
)
