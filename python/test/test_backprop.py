# -*- coding: utf-8 -*-
import numpy as np
import pytest
from celerite2 import backprop, driver
from celerite2.testing import get_matrices


def check_grad(fwd, rev, in_args, out_args, extra_args, eps=1.234e-8):
    vals0 = tuple(map(np.copy, fwd(*(in_args + out_args + extra_args))))

    # Compute numerical grad
    grads = []
    for n in range(len(in_args)):
        grads.append([np.empty((in_args[n].size, o.size)) for o in out_args])
        for m in range(in_args[n].size):
            in_args[n].flat[m] += eps
            vals = fwd(*(in_args + out_args + extra_args))
            in_args[n].flat[m] -= eps

            for k in range(len(out_args)):
                grads[n][k][m, :] = (vals[k] - vals0[k]).flatten() / eps

    # Make sure that everything is computed at the reference point
    fwd(*(in_args + out_args + extra_args))

    # Compute the backprop
    b_out = [np.zeros_like(a) for a in out_args]
    b_in = [np.zeros_like(a) for a in in_args]
    for k in range(len(out_args)):
        for i in range(out_args[k].size):
            # for b in b_in:
            #     b[:] = 0.0

            b_out[k].flat[i] = 1.0
            res = rev(*(in_args + out_args + extra_args + b_out + b_in))
            b_out[k].flat[i] = 0.0

            for n, b in enumerate(res):
                assert np.allclose(
                    b.flatten(), grads[n][k][:, i], atol=1e-3
                ), (k, i, n)


def test_factor_fwd():
    x, c, a, U, V, Y = get_matrices()

    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((len(a), U.shape[1], U.shape[1]))

    d0, W0 = driver.factor(x, c, a, U, V, np.copy(a), np.copy(V))
    d, W, S = backprop.factor_fwd(x, c, a, U, V, d, W, S)

    assert np.allclose(d, d0)
    assert np.allclose(W, W0)


def test_factor_rev():
    x, c, a, U, V, Y = get_matrices()

    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((len(a), U.shape[1], U.shape[1]))
    d, W, S = backprop.factor_fwd(x, c, a, U, V, d, W, S)

    check_grad(
        backprop.factor_fwd, backprop.factor_rev, [x, c, a, U, V], [d, W], [S]
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_lower_fwd(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    d, W = driver.factor(x, c, a, U, V, a, V)
    Z0 = driver.solve_lower(x, c, U, W, Y, np.copy(Y))

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.solve_lower_fwd(x, c, U, W, Y, Z, F)
    assert np.allclose(Z0, Z)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_lower_rev(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    d, W = driver.factor(x, c, a, U, V, a, V)

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.solve_lower_fwd(x, c, U, W, Y, Z, F)

    check_grad(
        backprop.solve_lower_fwd,
        backprop.solve_lower_rev,
        [x, c, U, W, Y],
        [Z],
        [F],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_upper_fwd(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    d, W = driver.factor(x, c, a, U, V, a, V)
    Z0 = driver.solve_upper(x, c, U, W, Y, np.copy(Y))

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.solve_upper_fwd(x, c, U, W, Y, Z, F)
    assert np.allclose(Z0, Z)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_upper_rev(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    d, W = driver.factor(x, c, a, U, V, a, V)

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.solve_upper_fwd(x, c, U, W, Y, Z, F)

    check_grad(
        backprop.solve_upper_fwd,
        backprop.solve_upper_rev,
        [x, c, U, W, Y],
        [Z],
        [F],
    )


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_lower_fwd(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    Z0 = driver.matmul_lower(x, c, U, V, Y, np.zeros_like(Y))

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.matmul_lower_fwd(x, c, U, V, Y, Z, F)
    assert np.allclose(Z0, Z)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_lower_rev(vector):
    x, c, a, U, V, Y = get_matrices(vector=vector)
    if vector:
        Y = Y[:, None]

    Z = np.empty_like(Y)
    F = np.empty((U.shape[0], U.shape[1], Y.shape[1]))
    Z, F = backprop.matmul_lower_fwd(x, c, U, V, Y, Z, F)

    check_grad(
        backprop.matmul_lower_fwd,
        backprop.matmul_lower_rev,
        [x, c, U, V, Y],
        [Z],
        [F],
    )
