# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import driver
from celerite2.testing import get_matrices

try:
    from jax.config import config
except ImportError:
    HAS_JAX = False
else:
    HAS_JAX = True

    config.update("jax_enable_x64", True)

    from jax.test_util import check_grads

    from celerite2.jax import ops

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="jax is not installed")


def check_op(op, input_arrays, expected_outputs, grad=True):
    output_tensors = op(*input_arrays)

    if len(expected_outputs) > 1:
        for array, tensor in zip(expected_outputs, output_tensors):
            assert np.allclose(array, np.asarray(tensor))
    else:
        assert np.allclose(expected_outputs[0], np.asarray(output_tensors))

    if grad:
        check_grads(op, input_arrays, 1, modes=["rev"], eps=1e-6)


def test_factor():
    a, U, V, P, Y = get_matrices()
    d, W = driver.factor(U, P, np.copy(a), np.copy(V))
    check_op(ops.factor, [a, U, V, P], [d, W])


@pytest.mark.parametrize("vector", [True, False])
def test_solve(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    X = driver.solve(U, P, d, W, np.copy(Y))
    check_op(ops.solve, [U, P, d, W, Y], [X])


def test_norm():
    a, U, V, P, Y = get_matrices(vector=True)
    d, W = driver.factor(U, P, a, V)
    X = driver.norm(U, P, d, W, np.copy(Y))
    check_op(ops.norm, [U, P, d, W, Y], [X])


@pytest.mark.parametrize("vector", [True, False])
def test_dot_tril(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    d, W = driver.factor(U, P, a, V)
    X = driver.dot_tril(U, P, d, W, np.copy(Y))
    check_op(ops.dot_tril, [U, P, d, W, Y], [X])


@pytest.mark.parametrize("vector", [True, False])
def test_matmul(vector):
    a, U, V, P, Y = get_matrices(vector=vector)
    X = driver.matmul(a, U, V, P, Y, np.copy(Y))
    check_op(ops.matmul, [a, U, V, P, Y], [X])


def test_conditional_mean():
    a, U, V, P, Y, U_star, V_star, inds = get_matrices(
        vector=True, conditional=True
    )
    d, W = driver.factor(U, P, a, np.copy(V))
    z = driver.solve(U, P, d, W, Y)

    mu = driver.conditional_mean(
        U, V, P, z, U_star, V_star, inds, np.empty(len(inds), dtype=np.float64)
    )

    check_op(
        ops.conditional_mean,
        [U, V, P, z, U_star, V_star, inds],
        [mu],
        grad=False,
    )
