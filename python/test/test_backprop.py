# -*- coding: utf-8 -*-
import numpy as np

from celerite2 import backprop, driver, terms


def get_matrices(size=100, kernel=None):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    Y = np.ascontiguousarray(
        np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
    )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    return a, U, V, P, Y


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
            for b in b_in:
                b[:] = 0.0

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
