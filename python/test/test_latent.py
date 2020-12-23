# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import latent, terms


@pytest.fixture
def data():
    N = 10
    M = 3
    np.random.seed(105)
    x = np.sort(np.random.uniform(0, 10, N))
    y = np.random.randn(N * M, 3)
    data = latent.prepare_rectangular_data(N, M, t=x)

    return N, M, x, data["t"], data["X"], y


def test_value(data):
    N, M, t0, t, X, y = data

    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)]) + 1e-5
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term += terms.SHOTerm(sigma=0.23456, rho=3.4, Q=2.3)
    kernel = latent.KroneckerLatentTerm(term, K=R)

    ar, cr, ac, bc, cc, dc = term.get_coefficients()
    c0, a0, U0, V0 = term.get_celerite_matrices(t0, np.zeros_like(t0))
    c, a, U, V = kernel.get_celerite_matrices(t, np.zeros_like(t), X=X)

    assert c.shape == (len(c0) * M,)
    assert a.shape == (N * M,)
    assert U.shape == (N * M, len(c0) * M)
    assert V.shape == (N * M, len(c0) * M)

    assert np.allclose(c, np.repeat(c0, M))
    assert np.allclose(a, (a0[:, None] * np.diag(R)[None, :]).flatten())
    assert np.allclose(U, np.kron(U0, R))
    assert np.allclose(V, np.kron(V0, np.eye(M)))

    K = kernel.get_value(t, X=X)
    K0 = term.get_value(t0)

    assert np.allclose(K, np.kron(K0, R))


def test_low_rank_value(data):
    N, M, t0, t, X, y = data

    alpha = np.random.randn(M)
    R = np.outer(alpha, alpha)
    term = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term += terms.SHOTerm(sigma=0.23456, rho=3.4, Q=2.3)
    kernel = latent.KroneckerLatentTerm(term, L=alpha)

    ar, cr, ac, bc, cc, dc = term.get_coefficients()
    c0, a0, U0, V0 = term.get_celerite_matrices(t0, np.zeros_like(t0))
    c, a, U, V = kernel.get_celerite_matrices(t, np.zeros_like(t), X=X)

    assert c.shape == (len(c0),)
    assert a.shape == (N * M,)
    assert U.shape == (N * M, len(c0))
    assert V.shape == (N * M, len(c0))

    assert np.allclose(c, c0)
    assert np.allclose(a, (a0[:, None] * np.diag(R)[None, :]).flatten())

    K = kernel.get_value(t, X=X)
    K0 = term.get_value(t0)

    assert np.allclose(K, np.kron(K0, R))
