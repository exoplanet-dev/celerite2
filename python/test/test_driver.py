# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import driver, terms
from celerite2.testing import get_matrices


def test_factor():
    a, U, V, P, K, Y = get_matrices(include_dense=True)
    d, W = driver.factor(U, P, a, V)

    # Make sure that no copy is made if possible
    assert np.allclose(a, d)
    assert np.allclose(V, W)


@pytest.mark.parametrize("vector", [True, False])
def test_solve(vector):
    a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

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
    a, U, V, P, K, Y = get_matrices(vector=True, include_dense=True)

    # First compute the expected value
    expect = np.dot(Y, np.linalg.solve(K, Y))

    # Then solve using celerite
    d, W = driver.factor(U, P, a, V)
    value = driver.norm(U, P, d, W, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul(vector):
    a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

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
        kernel=terms.RealTerm(a=1.0, c=0.5), vector=vector, include_dense=True
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
    a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

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
        vector=True, conditional=True, include_dense=True
    )

    # First compute the expected value
    alpha = np.linalg.solve(K, Y)
    expect = np.dot(K_star, alpha)

    # Then solve using celerite
    mu = np.empty(len(U_star))
    value = driver.conditional_mean(U, V, P, alpha, U_star, V_star, inds, mu)

    # Check that the solution is correct
    assert np.allclose(value, expect)
