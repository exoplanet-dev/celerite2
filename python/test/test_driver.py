# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import driver, terms
from celerite2.testing import get_matrices


def test_factor():
    x, c, a, U, V, K, Y = get_matrices(include_dense=True)
    d, W = driver.factor(x, c, U, a, V)

    # Make sure that no copy is made if possible
    assert np.allclose(a, d)
    assert np.allclose(V, W)


@pytest.mark.parametrize("vector", [True, False])
def test_solve(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    # First compute the expected value
    expect = np.linalg.solve(K, Y)

    # Then solve using celerite
    d, W = driver.factor(x, c, U, a, V)
    value = driver.solve(x, c, U, d, W, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


def test_norm():
    x, c, a, U, V, K, Y = get_matrices(vector=True, include_dense=True)

    # First compute the expected value
    expect = np.dot(Y, np.linalg.solve(K, Y))

    # Then solve using celerite
    d, W = driver.factor(x, c, U, a, V)
    value = driver.norm(x, c, U, d, W, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(x, c, a, U, V, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_order(vector):
    x, c, a, U, V, K, Y = get_matrices(
        kernel=terms.RealTerm(a=1.0, c=0.5), vector=vector, include_dense=True
    )

    # First compute the expected value
    expect = np.dot(K, Y)

    # Then solve using celerite
    Z = np.empty_like(Y)
    value = driver.matmul(x, c, a, U, V, Y, Z)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Z)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril(vector):
    x, c, a, U, V, K, Y = get_matrices(vector=vector, include_dense=True)

    # First compute the expected value
    expect = np.dot(np.linalg.cholesky(K), Y)

    # Then solve using celerite
    d, W = driver.factor(x, c, U, a, V)
    value = driver.dot_tril(x, c, U, d, W, Y)

    # Make sure that no copy is made if possible
    assert np.allclose(value, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_general_dot(vector):
    x, c, a, U, V, K, Y, t, U2, V2, K_star = get_matrices(
        conditional=True, include_dense=True, vector=vector
    )

    if vector:
        Z = np.zeros(len(t))
    else:
        Z = np.zeros((len(t), Y.shape[1]))
    Z = driver.general_lower_dot(t, x, c, U2, V, Y, Z)
    Z = driver.general_upper_dot(t, x, c, V2, U, Y, Z)

    assert np.allclose(np.dot(K_star, Y), Z)
