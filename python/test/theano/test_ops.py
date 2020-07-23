# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import backprop, driver
from celerite2.testing import get_matrices

try:
    import theano
except ImportError:
    HAS_THEANO = False
else:
    HAS_THEANO = True
    from theano import tensor as tt

    from celerite2.theano import ops


pytestmark = pytest.mark.skipif(
    not HAS_THEANO, reason="Theano is not installed"
)


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
    a, U, V, P, Y = get_matrices()
    check_basic(
        backprop.factor_fwd, ops.factor, [a, U, V, P],
    )


def test_factor_rev():
    a, U, V, P, Y = get_matrices()
    check_grad(
        ops.factor, [a, U, V, P], 2,
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_basic(
        backprop.solve_fwd, ops.solve, [U, P, d, W, Y],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_grad(
        ops.solve, [U, P, d, W, Y], 1,
    )


def test_norm_fwd():
    a, U, V, P, Y = get_matrices(vector=True)
    d, W = driver.factor(U, P, a, V)
    check_basic(
        backprop.norm_fwd, ops.norm, [U, P, d, W, Y],
    )


def test_norm_rev():
    a, U, V, P, Y = get_matrices(vector=True)
    d, W = driver.factor(U, P, a, V)
    check_grad(
        ops.norm, [U, P, d, W, Y], 1,
    )


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_basic(
        backprop.dot_tril_fwd, ops.dot_tril, [U, P, d, W, Y],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_grad(
        ops.dot_tril, [U, P, d, W, Y], 1,
    )


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    check_basic(
        backprop.matmul_fwd, ops.matmul, [a, U, V, P, Y],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    check_grad(
        ops.matmul, [a, U, V, P, Y], 1,
    )


def test_conditional_mean_fwd():
    a, U, V, P, Y, U_star, V_star, inds = get_matrices(
        vector=True, conditional=True
    )
    d, W = driver.factor(U, P, a, np.copy(V))
    z = driver.solve(U, P, d, W, Y)
    check_basic(
        driver.conditional_mean,
        ops.conditional_mean,
        [U, V, P, z, U_star, V_star, inds],
    )
