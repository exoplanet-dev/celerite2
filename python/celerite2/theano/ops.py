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
import json
from itertools import chain

import aesara_theano_fallback.tensor as tt
import numpy as np
import pkg_resources
from aesara_theano_fallback import aesara as theano
from aesara_theano_fallback import ifelse
from aesara_theano_fallback.graph import basic, op

from .. import backprop, driver


def _resize_or_set(outputs, n, shape):
    if outputs[n][0] is None:
        outputs[n][0] = np.empty(shape)
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
            tt.TensorType(
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
            tt.zeros_like(outputs[n])
            if isinstance(b.type, theano.gradient.DisconnectedType)
            else b
            for n, b in enumerate(gradients[: len(self.spec["outputs"])])
        )
        return self.rev_op(*chain(inputs, outputs, grads))


with open(
    pkg_resources.resource_filename("celerite2", "definitions.json"), "r"
) as f:
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
