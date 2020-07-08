# -*- coding: utf-8 -*-
import celerite as original_celerite
import numpy as np

from celerite2 import driver, terms


def get_celerite_matrices(kernel, x, diag):
    ar, cr, ac, bc, cc, dc = kernel.get_all_coefficients()
    a = diag + np.sum(ar) + np.sum(ac)

    N = len(x)
    Jr = len(ar)
    Jc = len(ac)
    J = Jr + 2 * Jc
    U = np.empty((N, J))
    V = np.empty((N, J))

    V[:, :Jr] = 1
    cos = np.cos(dc[None, :] * x[:, None], out=V[:, Jr : Jr + Jc])
    sin = np.sin(dc[None, :] * x[:, None], out=V[:, Jr + Jc :])

    U[:, :Jr] = ar[None, :]
    np.add(ac[None, :] * cos, bc[None, :] * sin, out=U[:, Jr : Jr + Jc])
    np.subtract(ac[None, :] * sin, bc[None, :] * cos, out=U[:, Jr + Jc :])

    dx = x[1:] - x[:-1]
    c = np.concatenate((cr, cc, cc))
    P = np.exp(-c[None, :] * dx[:, None])

    return a, U, V, P


def get_matrices(size=100, kernel=None):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    Y = np.ascontiguousarray(
        np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
    )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    K = kernel.get_value(x[:, None] - x[None, :])
    K[np.diag_indices_from(K)] += diag

    return a, U, V, P, K, Y


def test_factor():
    a, U, V, P, K, Y = get_matrices()
    d, W = driver.factor(U, P, a, V)

    # Make sure that no copy is made if possible
    assert np.allclose(a, d)
    assert np.allclose(V, W)


def test_solve():
    a, U, V, P, K, Y = get_matrices()

    # First compute the expected value
    expect = np.linalg.solve(K, Y)

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.solve(U, P, d, W, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_norm():
    a, U, V, P, K, Y = get_matrices()

    # First compute the expected value
    expect = np.dot(Y[:, 0], np.linalg.solve(K, Y[:, 0]))

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.norm(U, P, d, W, Y[:, 0])

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_matmul():
    a, U, V, P, K, Y = get_matrices()

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(a, U, V, P, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_matmul_order():
    a, U, V, P, K, Y = get_matrices(kernel=terms.RealTerm(a=1.0, c=0.5))

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(a, U, V, P, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_dot_tril():
    a, U, V, P, K, Y = get_matrices()

    # First compute the expected value
    expect = np.dot(np.linalg.cholesky(K), Y)

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.dot_tril(U, P, d, W, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)
