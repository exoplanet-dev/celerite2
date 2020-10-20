# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as tt

from celerite2.theano import terms, kron


def test_value():
    N = 100
    M = 5

    np.random.seed(105)
    x = np.sort(np.random.uniform(0, 10, N))
    t = np.sort(np.random.uniform(-1, 11, 25))
    diag = np.random.uniform(0.1, 0.5, (N, M))
    y = np.random.randn(N * M, 3)

    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, R=R)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.eval().shape == (N * M,)
    assert U.eval().shape == (N * M, 2 * M)
    assert V.eval().shape == (N * M, 2 * M)
    assert P.eval().shape == (N * M - 1, 2 * M)

    # check_value(term, x, diag, y, t)


def test_low_rank_value():
    N = 100
    M = 5

    np.random.seed(105)
    x = np.sort(np.random.uniform(0, 10, N))
    t = np.sort(np.random.uniform(-1, 11, 25))
    diag = np.random.uniform(0.1, 0.5, (N, M))
    y = np.random.randn(N * M, 3)

    alpha = np.random.randn(M)
    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.LowRankKronTerm(term0, alpha=alpha)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.eval().shape == (N * M,)
    assert U.eval().shape == (N * M, 2)
    assert V.eval().shape == (N * M, 2)
    assert P.eval().shape == (N * M - 1, 2)

    # check_value(term, x, diag, y, t)

    full_term = kron.KronTerm(term0, R=np.outer(alpha, alpha))
    assert np.allclose(
        full_term.dot(x, diag, y).eval(), term.dot(x, diag, y).eval()
    )
