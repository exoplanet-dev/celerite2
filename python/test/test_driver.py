# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import driver, terms


def get_matrices(size=100, kernel=None, vector=False, conditional=False):
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

    K = kernel.get_value(x[:, None] - x[None, :])
    K[np.diag_indices_from(K)] += diag

    if not conditional:
        return a, U, V, P, K, Y

    t = np.sort(np.random.uniform(-1, 12, 200))
    U_star, V_star, inds = kernel.get_conditional_mean_matrices(x, t)
    K_star = kernel.get_value(t[:, None] - x[None, :])
    return a, U, V, P, K, Y, U_star, V_star, inds, K_star


def test_factor():
    a, U, V, P, K, Y = get_matrices()
    d, W = driver.factor(U, P, a, V)

    # Make sure that no copy is made if possible
    assert np.allclose(a, d)
    assert np.allclose(V, W)


@pytest.mark.parametrize("vector", [True, False])
def test_solve(vector):
    a, U, V, P, K, Y = get_matrices(vector=vector)

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
    a, U, V, P, K, Y = get_matrices(vector=True)

    # First compute the expected value
    expect = np.dot(Y, np.linalg.solve(K, Y))

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.norm(U, P, d, W, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul(vector):
    a, U, V, P, K, Y = get_matrices(vector=vector)

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(a, U, V, P, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_order(vector):
    a, U, V, P, K, Y = get_matrices(
        kernel=terms.RealTerm(a=1.0, c=0.5), vector=vector
    )

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(a, U, V, P, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril(vector):
    a, U, V, P, K, Y = get_matrices(vector=vector)

    # First compute the expected value
    expect = np.dot(np.linalg.cholesky(K), Y)

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.dot_tril(U, P, d, W, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_conditional_mean():
    a, U, V, P, K, Y, U_star, V_star, inds, K_star = get_matrices(
        vector=True, conditional=True
    )

    # First compute the expected value
    alpha = np.linalg.solve(K, Y)
    expect = np.dot(K_star, alpha)

    # Then solve using celerite
    mu = np.empty(len(U_star))
    value = driver.conditional_mean(U, V, P, alpha, U_star, V_star, inds, mu)

    # Check that the solution is correct
    assert np.allclose(value, expect)
