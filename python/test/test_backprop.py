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
    a, U, V, P, Y = get_matrices()

    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((len(a), U.shape[1] ** 2))

    d0, W0 = driver.factor(U, P, np.copy(a), np.copy(V))
    d, W, S = backprop.factor_fwd(a, U, V, P, d, W, S)

    assert np.allclose(d, d0)
    assert np.allclose(W, W0)


def test_factor_rev():
    a, U, V, P, Y = get_matrices()

    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((len(a), U.shape[1] ** 2))
    d, W, S = backprop.factor_fwd(a, U, V, P, d, W, S)

    check_grad(
        backprop.factor_fwd, backprop.factor_rev, [a, U, V, P], [d, W], [S]
    )


@pytest.mark.parametrize("vector", [True, False])
def test_solve_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)

    X0 = driver.solve(U, P, d, W, np.copy(Y))

    X = np.empty_like(Y)
    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))
    G = np.empty_like(F)

    X, Z, F, G = backprop.solve_fwd(U, P, d, W, Y, X, Z, F, G)
    assert np.allclose(X0, X)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)

    X = np.empty_like(Y)
    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))
    G = np.empty_like(F)

    X, Z, F, G = backprop.solve_fwd(U, P, d, W, Y, X, Z, F, G)

    check_grad(
        backprop.solve_fwd, backprop.solve_rev, [U, P, d, W, Y], [X], [Z, F, G]
    )


def test_norm_fwd():
    a, U, V, P, Y = get_matrices(vector=True)
    d, W = driver.factor(U, P, a, V)

    X0 = driver.norm(U, P, d, W, np.copy(Y))

    X = np.empty((1, 1))
    Z = np.empty_like(Y)
    F = np.empty_like(U)

    X, Z, F = backprop.norm_fwd(U, P, d, W, Y, X, Z, F)
    assert np.allclose(X0, X)


def test_norm_rev():
    a, U, V, P, Y = get_matrices(vector=True)
    d, W = driver.factor(U, P, a, V)

    X = np.empty((1, 1))
    Z = np.empty_like(Y)
    F = np.empty_like(U)

    X, Z, F = backprop.norm_fwd(U, P, d, W, Y, X, Z, F)

    check_grad(
        backprop.norm_fwd, backprop.norm_rev, [U, P, d, W, Y], [X], [Z, F]
    )


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)

    Z0 = driver.dot_tril(U, P, d, W, np.copy(Y))

    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))

    Z, F = backprop.dot_tril_fwd(U, P, d, W, Y, Z, F)
    assert np.allclose(Z0, Z)


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)

    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))

    Z, F = backprop.dot_tril_fwd(U, P, d, W, Y, Z, F)

    check_grad(
        backprop.dot_tril_fwd, backprop.dot_tril_rev, [U, P, d, W, Y], [Z], [F]
    )


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_fwd(vector):
    a, U, V, P, Y = get_matrices(vector=vector)

    X0 = driver.matmul(a, U, V, P, Y, np.empty_like(Y))

    X = np.empty_like(Y)
    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))
    G = np.empty_like(F)

    X, Z, F, G = backprop.matmul_fwd(a, U, V, P, Y, X, Z, F, G)
    assert np.allclose(X0, X)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_rev(vector):
    a, U, V, P, Y = get_matrices(vector=vector)

    X = np.empty_like(Y)
    Z = np.empty_like(Y)
    if vector:
        F = np.empty_like(U)
    else:
        F = np.empty((U.shape[0], U.shape[1] * Y.shape[1]))
    G = np.empty_like(F)

    X, Z, F, G = backprop.matmul_fwd(a, U, V, P, Y, X, Z, F, G)

    check_grad(
        backprop.matmul_fwd,
        backprop.matmul_rev,
        [a, U, V, P, Y],
        [X],
        [Z, F, G],
    )
