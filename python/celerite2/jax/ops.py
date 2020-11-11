# -*- coding: utf-8 -*-

"""
Note: for better performance, these should be implemented as JAX primitives
with custom XLA calls:

- https://jax.readthedocs.io/en/stable/notebooks/How_JAX_primitives_work.html
- https://github.com/danieljtait/jax_xla_adventures
"""

__all__ = [
    "factor",
    "solve_lower",
    "solve_upper",
    "matmul_lower",
    "matmul_upper",
    "general_matmul_lower",
    "general_matmul_upper",
]
import json
from collections import OrderedDict
from functools import partial

import numpy as np
import pkg_resources
from jax import core, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, xla
from jax.lib import xla_client

from . import xla_ops

xops = xla_client.ops


def factor(t, c, a, U, V):
    d, W, S = factor_p.bind(t, c, a, U, V)
    return d, W


def solve_lower(t, c, U, W, Y):
    Z, F = solve_lower_p.bind(t, c, U, W, Y)
    return Z


def solve_upper(t, c, U, W, Y):
    Z, F = solve_upper_p.bind(t, c, U, W, Y)
    return Z


def matmul_lower(t, c, U, V, Y):
    Z, F = matmul_lower_p.bind(t, c, U, V, Y)
    return Z


def matmul_upper(t, c, U, V, Y):
    Z, F = matmul_upper_p.bind(t, c, U, V, Y)
    return Z


def general_matmul_lower(t1, t2, c, U, V, Y):
    Z, F = general_matmul_lower_p.bind(t1, t2, c, U, V, Y)
    return Z


def general_matmul_upper(t1, t2, c, U, V, Y):
    Z, F = general_matmul_upper_p.bind(t1, t2, c, U, V, Y)
    return Z


def _abstract_eval(spec, *args):
    if len(args) != len(spec["inputs"]):
        raise ValueError(
            f"{spec['name']} expected {len(spec['inputs'])} inputs, "
            f"got {len(args)}"
        )
    if any(arg.dtype != np.float64 for arg in args):
        raise ValueError(f"{spec['name']} requires float64 precision")
    dims = {
        s["name"]: args[s["coords"][0]].shape[s["coords"][1]]
        for s in spec["dimensions"]
    }
    for s, arg in zip(spec["inputs"], args):
        if arg.ndim != len(s["shape"]):
            raise ValueError(
                f"Incorrect number of dimensions for {s['name']}; "
                f"expected {len(s['shape'])}, got {arg.ndim}"
            )
        if arg.shape != tuple(dims[k] for k in s["shape"]):
            raise ValueError(
                f"Incorrect shape for {s['name']}; "
                f"expected {tuple(dims[k] for k in s['shape'])}, "
                f"got {arg.shape}"
            )
    return tuple(
        ShapedArray(tuple(dims[k] for k in s["shape"]), np.float64)
        for s in spec["outputs"] + spec["extra_outputs"]
    )


def _translation_rule(name, spec, c, *args):
    shapes = tuple(c.get_shape(arg) for arg in args)
    dims = OrderedDict(
        (s["name"], shapes[s["coords"][0]].dimensions()[s["coords"][1]])
        for s in spec["dimensions"]
    )
    if any(shape.element_type() != np.float64 for shape in shapes):
        raise ValueError(f"{spec['name']} requires float64 precision")

    return xops.CustomCallWithLayout(
        c,
        name,
        operands=tuple(
            xops.ConstantLiteral(c, np.int32(v)) for v in dims.values()
        )
        + args,
        shape_with_layout=xla_client.Shape.tuple_shape(
            tuple(
                xla_client.Shape.array_shape(
                    jnp.dtype(np.float64),
                    tuple(dims[k] for k in s["shape"]),
                    tuple(range(len(s["shape"]) - 1, -1, -1)),
                )
                for s in spec["outputs"] + spec["extra_outputs"]
            )
        ),
        operand_shapes_with_layout=tuple(
            xla_client.Shape.array_shape(jnp.dtype(jnp.int32), (), ())
            for _ in range(len(dims))
        )
        + tuple(
            xla_client.Shape.array_shape(
                jnp.dtype(np.float64),
                tuple(dims[k] for k in s["shape"]),
                tuple(range(len(s["shape"]) - 1, -1, -1)),
            )
            for s in spec["inputs"]
        ),
    )


def _jvp(prim, jvp_prim, spec, arg_values, arg_tangents):
    def make_zero(x, t):
        return lax.zeros_like_array(x) if type(t) is ad.Zero else t

    out_values = tuple(prim.bind(*arg_values))
    arg_tangents = tuple(
        make_zero(x, t) for x, t in zip(arg_values, arg_tangents)
    )
    out_tangents = tuple(
        jvp_prim.bind(*(arg_values + out_values + arg_tangents))
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


def _jvp_transpose(rev_prim, spec, c_out, *args):
    def make_zero(x, t):
        return lax.zeros_like_array(x) if type(t) is ad.Zero else t

    nin = len(spec["inputs"])
    nout = len(spec["outputs"])
    nargs = nin + nout + len(spec["extra_outputs"])
    c_out = tuple(
        make_zero(x, t) for x, t in zip(args[nin : nin + nout], c_out)
    )
    c_in = tuple(rev_prim.bind(*(args[:nargs] + c_out)))
    return tuple(None for _ in range(nargs)) + c_in


def _rev_abstract_eval(spec, *args):
    return tuple(
        ShapedArray(arg.shape, arg.dtype)
        for arg in args[: len(spec["inputs"])]
    )


def _rev_translation_rule(name, spec, c, *args):
    rev_spec = dict(
        name=f"{spec['name']}_rev",
        dimensions=spec["dimensions"],
        inputs=spec["inputs"]
        + spec["outputs"]
        + spec["extra_outputs"]
        + spec["outputs"],
        outputs=spec["inputs"],
        extra_outputs=[],
    )
    return _translation_rule(name, rev_spec, c, *args)


def _build_op(name, spec):
    xla_client.register_cpu_custom_call_target(
        name, getattr(xla_ops, spec["name"])()
    )

    prim = core.Primitive(f"celerite2_{spec['name']}")
    prim.multiple_results = True
    prim.def_impl(partial(xla.apply_primitive, prim))
    prim.def_abstract_eval(partial(_abstract_eval, spec))
    xla.backend_specific_translations["cpu"][prim] = partial(
        _translation_rule, name, spec
    )

    if not spec["has_rev"]:
        return prim

    xla_client.register_cpu_custom_call_target(
        name + b"_rev", getattr(xla_ops, f"{spec['name']}_rev")()
    )

    jvp_prim = core.Primitive(f"celerite2_{spec['name']}_jvp")
    jvp_prim.multiple_results = True
    rev_prim = core.Primitive(f"celerite2_{spec['name']}_rev")
    rev_prim.multiple_results = True

    # Setup a dummy JVP rule
    ad.primitive_jvps[prim] = partial(_jvp, prim, jvp_prim, spec)
    jvp_prim.def_abstract_eval(partial(_jvp_abstract_eval, spec))
    ad.primitive_transposes[jvp_prim] = partial(_jvp_transpose, rev_prim, spec)

    # Handle reverse pass using custom op
    rev_prim.def_impl(partial(xla.apply_primitive, rev_prim))
    rev_prim.def_abstract_eval(partial(_rev_abstract_eval, spec))
    xla.backend_specific_translations["cpu"][rev_prim] = partial(
        _rev_translation_rule, name + b"_rev", spec
    )

    return prim


with open(
    pkg_resources.resource_filename("celerite2", "definitions.json"), "r"
) as f:
    definitions = {spec["name"]: spec for spec in json.load(f)}


factor_p = _build_op(b"celerite2_factor", definitions["factor"])
solve_lower_p = _build_op(b"celerite2_solve_lower", definitions["solve_lower"])
solve_upper_p = _build_op(b"celerite2_solve_upper", definitions["solve_upper"])
matmul_lower_p = _build_op(
    b"celerite2_matmul_lower", definitions["matmul_lower"]
)
matmul_upper_p = _build_op(
    b"celerite2_matmul_upper", definitions["matmul_upper"]
)
general_matmul_lower_p = _build_op(
    b"celerite2_general_matmul_lower", definitions["general_matmul_lower"]
)
general_matmul_upper_p = _build_op(
    b"celerite2_general_matmul_upper", definitions["general_matmul_upper"]
)
