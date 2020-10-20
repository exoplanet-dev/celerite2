# -*- coding: utf-8 -*-
import numpy as np
import theano

from celerite2.theano import GaussianProcess, kron, terms


def check_value(term, x, diag, y, t):
    N, M = diag.shape

    K = term.get_value(x[:, None] - x[None, :]).eval()

    try:
        K0 = term.term.get_value(x[:, None] - x[None, :]).eval()
    except AttributeError:
        pass
    else:
        assert np.allclose(np.kron(K0, term.R.eval()), K)

    K += np.diag(diag.flatten())
    K0 = term.dot(x, diag, np.eye(diag.size)).eval()
    assert np.allclose(K, K0)
    assert np.allclose(np.dot(K, y), term.dot(x, diag, y).eval())

    gp = GaussianProcess(term, t=x, diag=diag)

    # "log_likelihood" method
    yval = y[:, 0].reshape((N, M))
    alpha = np.linalg.solve(K, y[:, 0])
    loglike = -0.5 * (
        np.dot(y[:, 0], alpha)
        + np.linalg.slogdet(K)[1]
        + len(K) * np.log(2 * np.pi)
    )
    assert np.allclose(loglike, gp.log_likelihood(yval).eval())

    # Predict
    K0 = K - np.diag(diag.flatten())
    mu0 = np.dot(K0, alpha)
    cov0 = K0 - np.dot(K0, np.linalg.solve(K, K0.T))

    mu, var = theano.function([], gp.predict(yval, return_var=True))()
    _, cov = theano.function([], gp.predict(yval, return_cov=True))()
    assert np.allclose(mu, mu0.reshape((N, M)))
    assert np.allclose(var, np.diag(cov0).reshape((N, M)))
    assert np.allclose(cov, cov0.reshape((N, M, N, M)))

    mu1, var1 = theano.function([], gp.predict(yval, t=x, return_var=True))()
    _, cov1 = theano.function([], gp.predict(yval, t=x, return_cov=True))()
    assert np.allclose(mu, mu1)
    assert np.allclose(var, var1)
    assert np.allclose(cov, cov1)

    K0 = term.get_value(t[:, None] - x[None, :]).eval()
    mu0 = np.dot(K0, alpha)
    cov0 = term.get_value(t[:, None] - t[None, :]).eval() - np.dot(
        K0, np.linalg.solve(K, K0.T)
    )
    mu, var = theano.function([], gp.predict(yval, t=t, return_var=True))()
    _, cov = theano.function([], gp.predict(yval, t=t, return_cov=True))()
    assert np.allclose(mu, mu0.reshape((len(t), M)))
    assert np.allclose(var, np.diag(cov0).reshape((len(t), M)))
    assert np.allclose(cov, cov0.reshape((len(t), M, len(t), M)))


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

    check_value(term, x, diag, y, t)


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

    check_value(term, x, diag, y, t)

    full_term = kron.KronTerm(term0, R=np.outer(alpha, alpha))
    assert np.allclose(
        full_term.dot(x, diag, y).eval(), term.dot(x, diag, y).eval()
    )


def test_sum_value():
    N = 100
    M = 5

    np.random.seed(105)
    x = np.sort(np.random.uniform(0, 10, N))
    t = np.sort(np.random.uniform(-1, 11, 25))
    diag = np.random.uniform(0.1, 0.5, (N, M))
    y = np.random.randn(N * M, 3)

    alpha = np.random.randn(M)
    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, R=R) + kron.LowRankKronTerm(term0, alpha=alpha)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.eval().shape == (N * M,)
    assert U.eval().shape == (N * M, 2 * M + 2)
    assert V.eval().shape == (N * M, 2 * M + 2)
    assert P.eval().shape == (N * M - 1, 2 * M + 2)

    check_value(term, x, diag, y, t)
