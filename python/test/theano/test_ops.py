# -*- coding: utf-8 -*-
import numpy as np
import pytest
import theano
from theano import tensor as tt

from celerite2 import backprop, driver, terms
from celerite2.theano import ops


def get_matrices(size=100, kernel=None, vector=False):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    if vector:
        Y = np.sin(x)
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
        )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    return a, U, V, P, Y


def convert_values_to_types(values):
    types = []
    for v in values:
        if v.ndim == 0:
            types.append(tt.dscalar())
        elif v.ndim == 1:
            types.append(tt.dvector())
        elif v.ndim == 2:
            types.append(tt.dmatrix())
        else:
            raise ValueError("unknown type")
    return types


def check_shape(op, inputs, values):
    outputs = op(*inputs)
    result = theano.function(inputs, outputs)(*values)
    shapes = theano.function(inputs, [o.shape for o in outputs])(*values)
    assert all(
        np.all(v.shape == s) for v, s in zip(result, shapes)
    ), "Invalid shape inference"


def check_basic(ref_func, op, values):
    inputs = convert_values_to_types(values)
    check_shape(op, inputs, values)
    outputs = op(*inputs)
    result = theano.function(inputs, outputs)(*values)

    expected = ref_func(*(values + [np.copy(r) for r in result]))

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
