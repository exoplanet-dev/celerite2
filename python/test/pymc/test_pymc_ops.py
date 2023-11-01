# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import pytest

from celerite2 import backprop, driver
from celerite2.testing import get_matrices

pytest.importorskip("celerite2.pymc")

try:
    import pytensor
    import pytensor.tensor as tt
    from pytensor.compile.mode import Mode
    from pytensor.compile.sharedvalue import SharedVariable
    from pytensor.graph.fg import FunctionGraph
    from pytensor.graph.rewriting.db import RewriteDatabaseQuery
    from pytensor.link.jax import JAXLinker

    from celerite2.pymc import ops
except (ImportError, ModuleNotFoundError):
    pass

try:
    import jax

except (ImportError, ModuleNotFoundError):
    jax = None

else:
    opts = RewriteDatabaseQuery(
        include=[None], exclude=["cxx_only", "BlasOpt"]
    )
    jax_mode = Mode(JAXLinker(), opts)
    py_mode = Mode("py", opts)


def convert_values_to_types(values):
    types = []
    for v in values:
        if v.dtype == "int64":
            types.append(tt.lvector())
        elif v.ndim == 0:
            types.append(tt.dscalar())
        elif v.ndim == 1:
            types.append(tt.dvector())
        elif v.ndim == 2:
            types.append(tt.dmatrix())
        else:
            raise ValueError("unknown type")
        types[-1].tag.test_value = v
    return types


def check_shape(op, inputs, outputs, values, result, multi):
    if multi:
        shapes = pytensor.function(inputs, [o.shape for o in outputs])(*values)
        assert all(
            np.all(v.shape == s) for v, s in zip(result, shapes)
        ), "Invalid shape inference"

    else:
        shape = pytensor.function(inputs, outputs.shape)(*values)
        assert result.shape == shape


def check_basic(ref_func, op, values):
    inputs = convert_values_to_types(values)
    outputs = op(*inputs)
    result = pytensor.function(inputs, outputs)(*values)

    try:
        result.shape
    except AttributeError:
        multi = True
        expected = ref_func(*(values + [np.copy(r) for r in result]))

    else:
        # We get here if there is only one output
        multi = False
        expected = ref_func(*(values + [np.copy(result)]))

    check_shape(op, inputs, outputs, values, result, multi)
    assert len(result) == len(expected)
    for a, b in zip(result, expected):
        np.testing.assert_allclose(a, b)

    if jax is not None:
        fg = FunctionGraph(inputs, outputs)
        compare_jax_and_py(fg, values)


def check_grad(op, values, num_out, eps=1.234e-8):
    inputs = convert_values_to_types(values)
    outputs = op(*inputs)
    func = pytensor.function(inputs, outputs)
    vals0 = func(*values)

    # Compute numerical grad
    grads = []
    for n in range(len(inputs)):
        grads.append(
            [np.empty((values[n].size, o.size)) for o in vals0[:num_out]]
        )
        for m in range(values[n].size):
            values[n].flat[m] += eps
            vals = func(*values)
            values[n].flat[m] -= eps

            for k in range(num_out):
                grads[n][k][m, :] = (vals[k] - vals0[k]).flatten() / eps

    # Compute the backprop
    for k in range(num_out):
        for i in range(vals0[k].size):
            res = pytensor.function(
                inputs, pytensor.grad(outputs[k].flatten()[i], inputs)
            )(*values)

            for n, b in enumerate(res):
                np.testing.assert_allclose(
                    b.flatten(), grads[n][k][:, i], atol=1e-3
                ), (k, i, n)

    if jax is not None:
        for k in range(num_out):
            out_grad = pytensor.grad(tt.sum(outputs[k]), inputs)
            fg = FunctionGraph(inputs, out_grad)
            compare_jax_and_py(fg, values)


def test_factor_fwd():
    x, c, a, U, V, Y = get_matrices()
    check_basic(
        backprop.factor_fwd,
        ops.factor,
        [x, c, a, U, V],
    )


def test_factor_rev():
    x, c, a, U, V, Y = get_matrices()
    check_grad(
        ops.factor,
        [x, c, a, U, V],
        2,
    )


def test_solve_lower_fwd():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, a, V)
    check_basic(
        backprop.solve_lower_fwd,
        ops.solve_lower,
        [x, c, U, W, Y],
    )


def test_solve_lower_rev():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, a, V)
    check_grad(
        ops.solve_lower,
        [x, c, U, W, Y],
        1,
    )


def test_solve_upper_fwd():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, a, V)
    check_basic(
        backprop.solve_upper_fwd,
        ops.solve_upper,
        [x, c, U, W, Y],
    )


def test_solve_upper_rev():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, a, V)
    check_grad(
        ops.solve_upper,
        [x, c, U, W, Y],
        1,
    )


def test_matmul_lower_fwd():
    x, c, a, U, V, Y = get_matrices()
    check_basic(
        backprop.matmul_lower_fwd,
        ops.matmul_lower,
        [x, c, U, V, Y],
    )


def test_matmul_lower_rev():
    x, c, a, U, V, Y = get_matrices()
    check_grad(
        ops.matmul_lower,
        [x, c, U, V, Y],
        1,
    )


def test_matmul_upper_fwd():
    x, c, a, U, V, Y = get_matrices()
    check_basic(
        backprop.matmul_upper_fwd,
        ops.matmul_upper,
        [x, c, U, V, Y],
    )


def test_matmul_upper_rev():
    x, c, a, U, V, Y = get_matrices()
    check_grad(
        ops.matmul_upper,
        [x, c, U, V, Y],
        1,
    )


def test_general_matmul_lower_fwd():
    x, c, a, U, V, Y, t, U2, V2 = get_matrices(conditional=True)
    check_basic(
        backprop.general_matmul_lower_fwd,
        ops.general_matmul_lower,
        [t, x, c, U2, V, Y],
    )


def test_general_matmul_upper_fwd():
    x, c, a, U, V, Y, t, U2, V2 = get_matrices(conditional=True)
    check_basic(
        backprop.general_matmul_upper_fwd,
        ops.general_matmul_upper,
        [t, x, c, U2, V, Y],
    )


def compare_jax_and_py(
    fgraph, test_inputs, assert_fn=None, must_be_device_array=True
):
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]
    pytensor_jax_fn = pytensor.function(
        fn_inputs, fgraph.outputs, mode=jax_mode
    )
    jax_res = pytensor_jax_fn(*test_inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(isinstance(res, jax.Array) for res in jax_res)
        else:
            assert isinstance(jax_res, jax.Array)

    pytensor_py_fn = pytensor.function(fn_inputs, fgraph.outputs, mode=py_mode)
    py_res = pytensor_py_fn(*test_inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(jax_res, py_res):
            print(np.min(j), np.max(j), np.any(np.isnan(j)))
            print(np.min(p), np.max(p), np.any(np.isnan(p)))

            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

    return jax_res
