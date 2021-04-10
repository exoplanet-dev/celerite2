# -*- coding: utf-8 -*-

import numpy as np
import pytest
from jax.config import config

from celerite2 import ops, terms

config.update("jax_enable_x64", True)


def get_matrices(
    size=100,
    kernel=None,
    vector=False,
    conditional=False,
    include_dense=False,
    no_diag=False,
):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    if vector:
        Y = np.sin(x)
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
        )
    if no_diag:
        diag = np.zeros_like(x)
    else:
        diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    if include_dense:
        K = kernel.get_value(x[:, None] - x[None, :])
        K += np.diag(diag)

    if not conditional:
        if include_dense:
            return x, a, U, V, P, K, Y
        return x, a, U, V, P, Y

    t = np.sort(np.random.uniform(-1, 12, 200))
    _, _, U2, V2 = kernel.get_celerite_matrices(t, np.zeros_like(t))

    if include_dense:
        K_star = kernel.get_value(t[:, None] - x[None, :])
        return x, a, U, V, P, K, Y, t, U2, V2, K_star

    return x, a, U, V, P, Y, t, U2, V2


@pytest.mark.parametrize("vector", [True, False])
def test_solve_lower(vector):
    x, a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.linalg.solve(np.linalg.cholesky(K), Y)

    # Then solve using celerite
    d, W = ops.factor(a, U, V, P)
    value = ops.solve_lower(U, W, P, Y)
    value /= np.sqrt(d)[:, None]

    # Check that the solution is correct
    np.testing.assert_allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_solve_upper(vector):
    x, a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.linalg.solve(np.linalg.cholesky(K).T, Y)

    # Then solve using celerite
    d, W = ops.factor(a, U, V, P)
    value = ops.solve_upper(U, W, P, Y / np.sqrt(d)[:, None])

    # Check that the solution is correct
    np.testing.assert_allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_lower(vector):
    x, a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(np.tril(K, -1), Y)

    # Then solve using celerite
    value = ops.matmul_lower(U, V, P, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)


@pytest.mark.parametrize("vector", [True, False])
def test_matmul_upper(vector):
    x, a, U, V, P, K, Y = get_matrices(vector=vector, include_dense=True)

    if vector:
        Y = Y[:, None]

    # First compute the expected value
    expect = np.dot(np.triu(K, 1), Y)

    # Then solve using celerite
    value = ops.matmul_upper(U, V, P, Y)

    # Check that the solution is correct
    assert np.allclose(value, expect)
