# -*- coding: utf-8 -*-
import aesara_theano_fallback.tensor as tt
import numpy as np
from aesara_theano_fallback import aesara as theano
from celerite2 import backprop, driver
from celerite2.testing import get_matrices
from celerite2.theano import ops


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
        shapes = theano.function(inputs, [o.shape for o in outputs])(*values)
        assert all(
            np.all(v.shape == s) for v, s in zip(result, shapes)
        ), "Invalid shape inference"

    else:
        shape = theano.function(inputs, outputs.shape)(*values)
        assert result.shape == shape


def check_basic(ref_func, op, values):
    inputs = convert_values_to_types(values)
    outputs = op(*inputs)
    result = theano.function(inputs, outputs)(*values)

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
        assert np.allclose(a, b)


def check_grad(op, values, num_out, eps=1.234e-8):
    inputs = convert_values_to_types(values)
    outputs = op(*inputs)
    func = theano.function(inputs, outputs)
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
            res = theano.function(
                inputs, theano.grad(outputs[k].flatten()[i], inputs)
            )(*values)

            for n, b in enumerate(res):
                assert np.allclose(
                    b.flatten(), grads[n][k][:, i], atol=1e-3
                ), (k, i, n)


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
