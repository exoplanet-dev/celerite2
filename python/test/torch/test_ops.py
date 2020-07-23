# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import driver, terms

try:
    import torch
    from torch.autograd import gradcheck

    from celerite2.torch import ops
except ImportError:
    HAS_TORCH = False
else:
    HAS_TORCH = True


pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch is not installed"
)


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

    if not conditional:
        return a, U, V, P, Y

    t = np.sort(np.random.uniform(-1, 12, 200))
    U_star, V_star, inds = kernel.get_conditional_mean_matrices(x, t)
    return a, U, V, P, Y, U_star, V_star, inds


def check_op(op, input_arrays, expected_outputs):
    input_tensors = tuple(
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in input_arrays
    )
    output_tensors = op(*input_tensors)

    if len(expected_outputs) > 1:
        for array, tensor in zip(expected_outputs, output_tensors):
            assert np.allclose(array, tensor.detach().numpy())
    else:
        assert np.allclose(
            expected_outputs[0], output_tensors.detach().numpy()
        )

    assert gradcheck(lambda *args: op(*args), input_tensors)


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
