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
        Y = np.sin(x)[:, None]
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
        )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    return a, U, V, P, Y


def check_grad(op, inputs, values, num_out, eps=1.234e-8):
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


def check_shape(op, inputs, values):
    outputs = op(*inputs)
    result = theano.function(inputs, outputs)(*values)
    shapes = theano.function(inputs, [o.shape for o in outputs])(*values)
    assert all(
        np.all(v.shape == s) for v, s in zip(result, shapes)
    ), "Invalid shape inference"


def check_basic(ref_func, op, inputs, values):
    check_shape(op, inputs, values)
    outputs = op(*inputs)
    result = theano.function(inputs, outputs)(*values)

    expected = ref_func(*(values + [np.copy(r) for r in result]))

    assert len(result) == len(expected)
    for a, b in zip(result, expected):
        assert np.allclose(a, b)


def test_factor_fwd():
    a, U, V, P, Y = get_matrices()
    check_basic(
        backprop.factor_fwd,
        ops.factor,
        [tt.dvector(), tt.dmatrix(), tt.dmatrix(), tt.dmatrix()],
        [a, U, V, P],
    )


def test_factor_rev():
    a, U, V, P, Y = get_matrices()
    check_grad(
        ops.factor,
        [tt.dvector(), tt.dmatrix(), tt.dmatrix(), tt.dmatrix()],
        [a, U, V, P],
        2,
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_basic(
        backprop.solve_fwd,
        ops.solve,
        [tt.dmatrix(), tt.dmatrix(), tt.dvector(), tt.dmatrix(), tt.dmatrix()],
        [U, P, d, W, Y],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    check_grad(
        ops.solve,
        [tt.dmatrix(), tt.dmatrix(), tt.dvector(), tt.dmatrix(), tt.dmatrix()],
        [U, P, d, W, Y],
        1,
    )
