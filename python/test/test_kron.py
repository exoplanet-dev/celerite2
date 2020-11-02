# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import GaussianProcess, kron, terms


@pytest.fixture
def data():
    N = 100
    M = 5
    np.random.seed(105)
    x = np.sort(np.random.uniform(0, 10, N))
    t = np.sort(np.random.uniform(-1, 11, 25))
    diag = np.random.uniform(0.1, 0.5, (N, M))
    y = np.random.randn(N * M, 3)
    return N, M, x, diag, y, t


def check_value(term, x, diag, y, t, mask=None):
    N, M = diag.shape

    K = term.get_value(x[:, None] - x[None, :])
    try:
        K0 = term.term.get_value(x[:, None] - x[None, :])
    except AttributeError:
        pass
    else:
        assert np.allclose(np.kron(K0, term.R), K)

    mask0 = mask
    if mask is not None:
        K00 = K[:, mask.flatten()]
        K = K00[mask.flatten(), :]
    else:
        K00 = np.copy(K)
        mask = np.ones_like(diag, dtype=bool)

    K[np.diag_indices_from(K)] += diag.flatten()[mask.flatten()]

    K0 = term.dot(x, diag, np.eye(mask.sum()), mask=mask)
    assert np.allclose(K, K0)
    assert np.allclose(
        np.dot(K, y[mask.flatten()]),
        term.dot(x, diag, y[mask.flatten()], mask=mask),
    )

    gp = GaussianProcess(term, t=x, diag=diag, mask=mask0)

    # "log_likelihood" method
    yval = y[:, 0].reshape((N, M))
    alpha = np.linalg.solve(K, y[mask.flatten(), 0])
    loglike = -0.5 * (
        np.dot(y[mask.flatten(), 0], alpha)
        + np.linalg.slogdet(K)[1]
        + len(K) * np.log(2 * np.pi)
    )
    assert np.allclose(loglike, gp.log_likelihood(yval))

    # Predict
    K0 = K00
    mu0 = np.dot(K0, alpha)
    cov0 = term.get_value(x[:, None] - x[None, :]) - np.dot(
        K0, np.linalg.solve(K, K0.T)
    )

    mu, var = gp.predict(yval, return_var=True)
    _, cov = gp.predict(yval, return_cov=True)
    assert mu.shape == (N, M)
    assert var.shape == (N, M)
    assert np.allclose(mu.flatten(), mu0)
    assert np.allclose(var.flatten(), np.diag(cov0))
    assert np.allclose(cov, cov0.reshape((N, M, N, M)))

    mu1, var1 = gp.predict(yval, t=x, return_var=True)
    _, cov1 = gp.predict(yval, t=x, return_cov=True)
    assert np.allclose(mu, mu1)
    assert np.allclose(var, var1)
    assert np.allclose(cov, cov1)

    K0 = term.get_value(t[:, None] - x[None, :])[:, mask.flatten()]
    mu0 = np.dot(K0, alpha)
    cov0 = term.get_value(t[:, None] - t[None, :]) - np.dot(
        K0, np.linalg.solve(K, K0.T)
    )
    mu, var = gp.predict(yval, t=t, return_var=True)
    _, cov = gp.predict(yval, t=t, return_cov=True)
    assert np.allclose(mu, mu0.reshape((len(t), M)))
    assert np.allclose(var, np.diag(cov0).reshape((len(t), M)))
    assert np.allclose(cov, cov0.reshape((len(t), M, len(t), M)))

    # "sample" method
    seed = 5938
    np.random.seed(seed)
    a = np.dot(np.linalg.cholesky(K), np.random.randn(len(K)))
    np.random.seed(seed)
    b = gp.sample()
    assert np.allclose(a, b[mask])

    np.random.seed(seed)
    a = np.dot(np.linalg.cholesky(K), np.random.randn(len(K), 10))
    np.random.seed(seed)
    b = gp.sample(size=10)
    assert np.allclose(
        np.moveaxis(a, -1, 0),
        b[:, mask],
    )

    # "sample_conditional" method, numerics make this one a little unstable;
    # just check the shape
    b = gp.sample_conditional(yval, t=t)
    assert b.shape == (len(t), M)

    b = gp.sample_conditional(yval, t=t, size=10)
    assert b.shape == (10, len(t), M)


def test_value(data):
    N, M, x, diag, y, t = data

    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, R=R)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.shape == (N * M,)
    assert U.shape == (N * M, 2 * M)
    assert V.shape == (N * M, 2 * M)
    assert P.shape == (N * M - 1, 2 * M)

    check_value(term, x, diag, y, t)


def test_low_rank_value(data):
    N, M, x, diag, y, t = data

    alpha = np.random.randn(M)
    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, L=alpha)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.shape == (N * M,)
    assert U.shape == (N * M, 2)
    assert V.shape == (N * M, 2)
    assert P.shape == (N * M - 1, 2)

    check_value(term, x, diag, y, t)

    full_term = kron.KronTerm(term0, R=np.outer(alpha, alpha))
    assert np.allclose(full_term.dot(x, diag, y), term.dot(x, diag, y))


def test_sum_value(data):
    N, M, x, diag, y, t = data

    alpha = np.random.randn(M)
    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, R=R) + kron.KronTerm(term0, L=alpha)

    a, U, V, P = term.get_celerite_matrices(x, diag)
    assert a.shape == (N * M,)
    assert U.shape == (N * M, 2 * M + 2)
    assert V.shape == (N * M, 2 * M + 2)
    assert P.shape == (N * M - 1, 2 * M + 2)

    check_value(term, x, diag, y, t)


def test_missing_values(data):
    N, M, x, diag, y, t = data
    mask = np.random.rand(N, M) > 0.1
    assert np.all(mask.sum(axis=1) > 0)

    R = np.random.randn(M, M)
    R[np.diag_indices_from(R)] = np.exp(R[np.diag_indices_from(R)])
    R[np.triu_indices_from(R, 1)] = 0.0
    R = np.dot(R, R.T)

    term0 = terms.SHOTerm(sigma=1.5, rho=1.3, Q=0.3)
    term = kron.KronTerm(term0, R=R)

    a, U, V, P = term.get_celerite_matrices(x, diag, mask=mask)
    assert a.shape == (mask.sum(),)
    assert U.shape == (mask.sum(), 2 * M)
    assert V.shape == (mask.sum(), 2 * M)
    assert P.shape == (mask.sum() - 1, 2 * M)

    check_value(term, x, diag, y, t, mask=mask)
