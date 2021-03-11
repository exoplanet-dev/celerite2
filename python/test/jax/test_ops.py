# -*- coding: utf-8 -*-
import numpy as np
from celerite2 import driver
from celerite2.jax import ops
from celerite2.testing import get_matrices
from jax import jit
from jax.test_util import check_grads


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
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, np.copy(a), np.copy(V))
    check_op(ops.factor, [x, c, a, U, V], [d, W])
    check_op(jit(ops.factor), [x, c, a, U, V], [d, W])


def test_solve_lower():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, np.copy(a), np.copy(V))
    Z = driver.solve_lower(x, c, U, W, Y, np.zeros_like(Y))
    check_op(ops.solve_lower, [x, c, U, W, Y], [Z])
    check_op(jit(ops.solve_lower), [x, c, U, W, Y], [Z])


def test_solve_upper():
    x, c, a, U, V, Y = get_matrices()
    d, W = driver.factor(x, c, a, U, V, np.copy(a), np.copy(V))
    Z = driver.solve_upper(x, c, U, W, Y, np.zeros_like(Y))
    check_op(ops.solve_upper, [x, c, U, W, Y], [Z])
    check_op(jit(ops.solve_upper), [x, c, U, W, Y], [Z])


def test_matmul_lower():
    x, c, a, U, V, Y = get_matrices()
    Z = driver.matmul_lower(x, c, U, V, Y, np.zeros_like(Y))
    check_op(ops.matmul_lower, [x, c, U, V, Y], [Z])
    check_op(jit(ops.matmul_lower), [x, c, U, V, Y], [Z])


def test_matmul_upper():
    x, c, a, U, V, Y = get_matrices()
    Z = driver.matmul_upper(x, c, U, V, Y, np.zeros_like(Y))
    check_op(ops.matmul_upper, [x, c, U, V, Y], [Z])
    check_op(jit(ops.matmul_upper), [x, c, U, V, Y], [Z])


def test_general_matmul_lower():
    x, c, a, U, V, Y, t, U2, V2 = get_matrices(conditional=True)
    Z = driver.general_matmul_lower(
        t, x, c, U2, V, Y, np.zeros((len(t), Y.shape[1]))
    )
    check_op(ops.general_matmul_lower, [t, x, c, U2, V, Y], [Z], grad=False)
    check_op(
        jit(ops.general_matmul_lower), [t, x, c, U2, V, Y], [Z], grad=False
    )


def test_general_matmul_upper():
    x, c, a, U, V, Y, t, U2, V2 = get_matrices(conditional=True)
    Z = driver.general_matmul_upper(
        t, x, c, U2, V, Y, np.zeros((len(t), Y.shape[1]))
    )
    check_op(ops.general_matmul_upper, [t, x, c, U2, V, Y], [Z], grad=False)
    check_op(
        jit(ops.general_matmul_upper), [t, x, c, U2, V, Y], [Z], grad=False
    )
