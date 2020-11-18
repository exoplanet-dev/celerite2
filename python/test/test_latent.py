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
    kernel = latent.KroneckerLatentTerm(term, R=R)

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


# def test_low_rank_value(data):
#     N, M, x, diag, y, t = data

#     alpha = np.random.randn(M)
#     term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
#     term = kron.KronTerm(term0, L=alpha)

#     a, U, V, P = term.get_celerite_matrices(x, diag)
#     assert a.shape == (N * M,)
#     assert U.shape == (N * M, 2)
#     assert V.shape == (N * M, 2)
#     assert P.shape == (N * M - 1, 2)

#     check_value(term, x, diag, y, t)

#     full_term = kron.KronTerm(term0, R=np.outer(alpha, alpha))
#     assert np.allclose(full_term.dot(x, diag, y), term.dot(x, diag, y))


# def test_sum_value(data):
#     N, M, x, diag, y, t = data

#     alpha = np.random.randn(M)
#     R = np.random.randn(M, M)
#     R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
#     R[np.triu_indices_from(R, 1)] = 0.0
#     R = np.dot(R, R.T)

#     term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
#     term = kron.KronTerm(term0, R=R) + kron.KronTerm(term0, L=alpha)

#     a, U, V, P = term.get_celerite_matrices(x, diag)
#     assert a.shape == (N * M,)
#     assert U.shape == (N * M, 2 * M + 2)
#     assert V.shape == (N * M, 2 * M + 2)
#     assert P.shape == (N * M - 1, 2 * M + 2)

#     check_value(term, x, diag, y, t)


# def test_missing_values(data):
#     N, M, x, diag, y, t = data
#     mask = np.random.rand(N, M) > 0.1
#     assert np.all(mask.sum(axis=1) > 0)

#     R = np.random.randn(M, M)
#     R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
#     R[np.triu_indices_from(R, 1)] = 0.0
#     R = np.dot(R, R.T)

#     term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
#     term = kron.KronTerm(term0, R=R)

#     a, U, V, P = term.get_celerite_matrices(x, diag, mask=mask)
#     assert a.shape == (mask.sum(),)
#     assert U.shape == (mask.sum(), 2 * M)
#     assert V.shape == (mask.sum(), 2 * M)
#     assert P.shape == (mask.sum() - 1, 2 * M)

#     check_value(term, x, diag, y, t, mask=mask)
