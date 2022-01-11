# -*- coding: utf-8 -*-
import numpy as np
import pytest
from celerite2 import driver
from celerite2.testing import get_matrices


def test_factor():
    x, c, a, U, V, K, Y = get_matrices(include_dense=True)
    d, W = driver.factor(x, c, a, U, V, a, V)

    # Make sure that no copy is made if possible
    assert np.allclose(a, d)
    assert np.allclose(V, W)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_lower(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.linalg.solve(np.linalg.cholesky(K), Y)

    # Then solve using celerite
    d, W = driver.factor(x, c, a, U, V, a, V)
    value = driver.solve_lower(x, c, U, W, Y, Y)
    Y /= np.sqrt(d)[:, None]

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_upper(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.linalg.solve(np.linalg.cholesky(K).T, Y)

    # Then solve using celerite
    d, W = driver.factor(x, c, a, U, V, a, V)
    Y /= np.sqrt(d)[:, None]
    value = driver.solve_upper(x, c, U, W, Y, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_lower(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(np.tril(K, -1), Y)

    # Then solve using celerite
    value = driver.matmul_lower(x, c, U, V, Y, np.zeros_like(Y))

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_upper(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(np.triu(K, 1), Y)

    # Then solve using celerite
    value = driver.matmul_upper(x, c, U, V, Y, np.zeros_like(Y))

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_general_matmul(vector):
    x, c, a, U, V, K, Y, t, U2, V2, K_star = get_matrices(
        conditional=True, include_dense=True, vector=vector
    )

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(K_star, Y)

    # Then solve using celerite
    Z = np.zeros((len(t), Y.shape[1]))
    Z = driver.general_matmul_lower(t, x, c, U2, V, Y, Z)
    Z = driver.general_matmul_upper(t, x, c, V2, U, Y, Z)

    # Check that the solution is correct
    assert np.allclose(Z, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_general_matmul_fallback(vector):
    x, c, a, U, V, K, Y = get_matrices(
        vector=vector, include_dense=True, no_diag=True
    )

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.zeros_like(Y)
    Z = driver.general_matmul_lower(x, x, c, U, V, Y, Z)
    Z = driver.general_matmul_upper(x, x, c, V, U, Y, Z)

    # Check that the solution is correct
    assert np.allclose(Z, expect)
