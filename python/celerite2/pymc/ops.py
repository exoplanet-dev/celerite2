# -*- coding: utf-8 -*-

__all__ = [
    "factor",
    "factor_quiet",
    "solve_lower",
    "solve_upper",
    "matmul_lower",
    "matmul_upper",
    "general_matmul_lower",
    "general_matmul_upper",
]

import importlib.resources as resources
import json
from itertools import chain

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import basic, op
from pytensor.link.jax.dispatch import jax_funcify

import celerite2.backprop as backprop
import celerite2.driver as driver


def _resize_or_set(outputs, n, shape):
    if outputs[n][0] is None:
        outputs[n][0] = np.zeros(shape)
    else:
        outputs[n][0] = np.ascontiguousarray(
            np.resize(outputs[n][0], shape), dtype=np.float64
        )
    return outputs[n][0]


class _CeleriteOp(op.Op):
    __props__ = ("name", "quiet")

    def __init__(self, name, spec, *, quiet=False):
        self.name = name
        self.spec = spec
        self.quiet = quiet

        if spec.get("has_rev", False):
            self.rev_op = _CeleriteOp(
                spec["name"] + "_rev",
                dict(
                    name=spec["name"] + "_rev",
                    has_rev=False,
                    dimensions=spec["dimensions"],
                    inputs=spec["inputs"]
                    + spec["outputs"]
                    + spec["extra_outputs"]
                    + spec["outputs"],
                    outputs=spec["inputs"],
                    extra_outputs=[],
                ),
            )

        super().__init__()

    def make_node(self, *inputs):
        if len(inputs) != len(self.spec["inputs"]):
            raise ValueError(
                f"{self.name} expected {len(self.spec['inputs'])} inputs, "
                f"got {len(inputs)}"
            )
        if any(arg.dtype != "float64" for arg in inputs):
            raise ValueError(f"{self.name} requires float64 precision")
        for arg, spec in zip(inputs, self.spec["inputs"]):
            if arg.ndim != len(spec["shape"]):
                raise ValueError(
                    f"Incorrect number of dimensions for {spec['name']}; "
                    f"expected {len(spec['shape'])}, got {arg.ndim}"
                )

        broadcastable = {
            spec["name"]: inputs[spec["coords"][0]].broadcastable[
                spec["coords"][1]
            ]
            for spec in self.spec["dimensions"]
        }
        otypes = [
            pt.TensorType(
                "float64", [broadcastable[k] for k in spec["shape"]]
            )()
            for spec in self.spec["outputs"] + self.spec["extra_outputs"]
        ]
        return basic.Apply(self, inputs, otypes)

    def infer_shape(self, *args):
        shapes = args[-1]
        dims = {
            spec["name"]: shapes[spec["coords"][0]][spec["coords"][1]]
            for spec in self.spec["dimensions"]
        }
        return tuple(
            tuple(dims[name] for name in spec["shape"])
            for spec in self.spec["outputs"] + self.spec["extra_outputs"]
        )

    def perform(self, node, inputs, outputs):
        dims = {
            spec["name"]: inputs[spec["coords"][0]].shape[spec["coords"][1]]
            for spec in self.spec["dimensions"]
        }
        args = tuple(
            np.ascontiguousarray(arg, dtype=np.float64) for arg in inputs
        ) + tuple(
            _resize_or_set(
                outputs, n, tuple(dims[name] for name in spec["shape"])
            )
            for n, spec in enumerate(
                self.spec["outputs"] + self.spec["extra_outputs"]
            )
        )
        try:
            func = getattr(backprop, self.name)
        except AttributeError:
            func = getattr(driver, self.name)

        try:
            func(*args)
        except backprop.LinAlgError:
            if not self.quiet:
                raise
            outputs[0][0][:] = -1.0

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        grads = (
            (
                pt.zeros_like(outputs[n])
                if isinstance(b.type, pytensor.gradient.DisconnectedType)
                else b
            )
            for n, b in enumerate(gradients[: len(self.spec["outputs"])])
        )
        return self.rev_op(*chain(inputs, outputs, grads))


with resources.files("celerite2").joinpath("definitions.json").open("r") as f:
    definitions = {spec["name"]: spec for spec in json.load(f)}


factor = _CeleriteOp("factor_fwd", definitions["factor"])
factor_quiet = _CeleriteOp("factor_fwd", definitions["factor"], quiet=True)
solve_lower = _CeleriteOp("solve_lower_fwd", definitions["solve_lower"])
solve_upper = _CeleriteOp("solve_upper_fwd", definitions["solve_upper"])
matmul_lower = _CeleriteOp("matmul_lower_fwd", definitions["matmul_lower"])
matmul_upper = _CeleriteOp("matmul_upper_fwd", definitions["matmul_upper"])
general_matmul_lower = _CeleriteOp(
    "general_matmul_lower_fwd", definitions["general_matmul_lower"]
)
general_matmul_upper = _CeleriteOp(
    "general_matmul_upper_fwd", definitions["general_matmul_upper"]
)


# JAX conversion for PyTensor JAX linker -------------------------------------
@jax_funcify.register(_CeleriteOp)
def _jax_funcify_celerite(op, node, **kwargs):
    """Map celerite2 PyTensor ops to their JAX counterparts."""

    # Lazy import to avoid circular import during module import
    import celerite2.jax.ops as jax_ops

    def factor_fwd(t, c, a, U, V):
        return jax_ops.factor_p.bind(t, c, a, U, V)

    def factor_rev(t, c, a, U, V, d, W, S, bd, bW):
        return jax_ops.factor_rev_p.bind(t, c, a, U, V, d, W, S, bd, bW)

    def solve_lower_fwd(t, c, U, W, Y):
        return jax_ops.solve_lower_p.bind(t, c, U, W, Y)

    def solve_lower_rev(t, c, U, W, Y, Z, F, bZ):
        return jax_ops.solve_lower_rev_p.bind(t, c, U, W, Y, Z, F, bZ)

    def solve_upper_fwd(t, c, U, W, Y):
        return jax_ops.solve_upper_p.bind(t, c, U, W, Y)

    def solve_upper_rev(t, c, U, W, Y, Z, F, bZ):
        return jax_ops.solve_upper_rev_p.bind(t, c, U, W, Y, Z, F, bZ)

    def matmul_lower_fwd(t, c, U, V, Y):
        return jax_ops.matmul_lower_p.bind(t, c, U, V, Y)

    def matmul_lower_rev(t, c, U, V, Y, Z, F, bZ):
        return jax_ops.matmul_lower_rev_p.bind(t, c, U, V, Y, Z, F, bZ)

    def matmul_upper_fwd(t, c, U, V, Y):
        return jax_ops.matmul_upper_p.bind(t, c, U, V, Y)

    def matmul_upper_rev(t, c, U, V, Y, Z, F, bZ):
        return jax_ops.matmul_upper_rev_p.bind(t, c, U, V, Y, Z, F, bZ)

    def general_matmul_lower_fwd(t1, t2, c, U, V, Y):
        return jax_ops.general_matmul_lower_p.bind(t1, t2, c, U, V, Y)

    def general_matmul_upper_fwd(t1, t2, c, U, V, Y):
        return jax_ops.general_matmul_upper_p.bind(t1, t2, c, U, V, Y)

    mapping = {
        "factor_fwd": factor_fwd,
        "factor_rev": factor_rev,
        "solve_lower_fwd": solve_lower_fwd,
        "solve_lower_rev": solve_lower_rev,
        "solve_upper_fwd": solve_upper_fwd,
        "solve_upper_rev": solve_upper_rev,
        "matmul_lower_fwd": matmul_lower_fwd,
        "matmul_lower_rev": matmul_lower_rev,
        "matmul_upper_fwd": matmul_upper_fwd,
        "matmul_upper_rev": matmul_upper_rev,
        "general_matmul_lower_fwd": general_matmul_lower_fwd,
        "general_matmul_upper_fwd": general_matmul_upper_fwd,
    }

    try:
        return mapping[op.name]
    except KeyError:
        raise NotImplementedError(
            f"No JAX conversion registered for {op.name}"
        )
