# -*- coding: utf-8 -*-
import celerite as original_celerite
import numpy as np
import pytest
from celerite import terms as cterms

import celerite2
from celerite2 import terms


@pytest.fixture
def data():
    # Generate fake data
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.sort(np.random.uniform(-1, 12, 100))
    diag = np.random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)
    return x, diag, y, t


test_terms = [
    cterms.RealTerm(log_a=np.log(2.5), log_c=np.log(1.1123)),
    cterms.RealTerm(log_a=np.log(12.345), log_c=np.log(1.5))
    + cterms.RealTerm(log_a=np.log(0.5), log_c=np.log(1.1234)),
    cterms.ComplexTerm(
        log_a=np.log(10.0), log_c=np.log(5.6), log_d=np.log(2.1)
    ),
    cterms.ComplexTerm(
        log_a=np.log(7.435),
        log_b=np.log(0.5),
        log_c=np.log(1.102),
        log_d=np.log(1.05),
    ),
    cterms.SHOTerm(
        log_S0=np.log(1.1), log_Q=np.log(0.1), log_omega0=np.log(1.2)
    ),
    cterms.SHOTerm(
        log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
    ),
    cterms.SHOTerm(
        log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
    )
    + cterms.RealTerm(log_a=np.log(1.345), log_c=np.log(2.4)),
    cterms.SHOTerm(
        log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
    )
    * cterms.RealTerm(log_a=np.log(1.345), log_c=np.log(2.4)),
    cterms.Matern32Term(log_sigma=0.1, log_rho=0.4),
]


@pytest.mark.parametrize("oterm", test_terms)
@pytest.mark.parametrize("mean", [0.0, 10.5])
def test_consistency(oterm, mean, data):
    x, diag, y, t = data

    # Setup the original GP
    original_gp = original_celerite.GP(oterm, mean=mean)
    original_gp.compute(x, np.sqrt(diag))

    # Setup the new GP
    term = terms.OriginalCeleriteTerm(oterm)
    gp = celerite2.GaussianProcess(term, mean=mean)
    gp.compute(x, diag=diag)

    # "log_likelihood" method
    assert np.allclose(original_gp.log_likelihood(y), gp.log_likelihood(y))

    # "predict" method
    for args in [
        dict(return_cov=False, return_var=False),
        dict(return_cov=False, return_var=True),
        dict(return_cov=True, return_var=False),
    ]:
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                original_gp.predict(y, **args),
                gp.predict(y, **args),
            )
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                original_gp.predict(y, t=t, **args),
                gp.predict(y, t=t, **args),
            )
        )

    # "sample" method
    seed = 5938
    np.random.seed(seed)
    a = original_gp.sample()
    np.random.seed(seed)
    b = gp.sample()
    assert np.allclose(a, b)

    np.random.seed(seed)
    a = original_gp.sample(size=10)
    np.random.seed(seed)
    b = gp.sample(size=10)
    assert np.allclose(a, b)

    # "sample_conditional" method, numerics make this one a little unstable;
    # just check the shape
    a = original_gp.sample_conditional(y, t=t)
    b = gp.sample_conditional(y, t=t)
    assert a.shape == b.shape

    a = original_gp.sample_conditional(y, size=10)
    b = gp.sample_conditional(y, size=10)
    assert a.shape == b.shape


def test_diag(data):
    x, diag, y, t = data

    term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
    gp1 = celerite2.GaussianProcess(term, t=x, diag=diag)
    gp2 = celerite2.GaussianProcess(term, t=x, yerr=np.sqrt(diag))
    assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))

    gp1 = celerite2.GaussianProcess(term, t=x, diag=np.zeros_like(x))
    gp2 = celerite2.GaussianProcess(term, t=x)
    assert np.allclose(gp1.log_likelihood(y), gp2.log_likelihood(y))


def test_mean(data):
    x, diag, y, t = data

    term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
    gp = celerite2.GaussianProcess(term, t=x, diag=diag, mean=lambda x: 2 * x)

    mu1 = gp.predict(y, include_mean=True)
    mu2 = gp.predict(y, x, include_mean=True)
    assert np.allclose(mu1, mu2)

    mu1 = gp.predict(y, include_mean=False)
    mu2 = gp.predict(y, x, include_mean=False)
    assert np.allclose(mu1, mu2)

    mu_mean = gp.predict(y, include_mean=True)
    mu_no_mean = gp.predict(y, include_mean=False)
    assert np.allclose(mu_mean, mu_no_mean + 2 * x)

    mu_mean = gp.predict(y, t, include_mean=True)
    mu_no_mean = gp.predict(y, t, include_mean=False)
    assert np.allclose(mu_mean, mu_no_mean + 2 * t)

    np.random.seed(42)
    s1 = gp.sample(size=5, include_mean=True)
    np.random.seed(42)
    s2 = gp.sample(size=5, include_mean=False)
    assert np.allclose(s1, s2 + 2 * x)


def test_predict_kernel(data):
    x, diag, y, t = data

    term1 = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
    term2 = terms.SHOTerm(S0=0.3, w0=0.1, Q=0.1)
    term = term1 + term2
    gp = celerite2.GaussianProcess(term, t=x, diag=diag)

    mu0 = gp.predict(y)
    mu1 = gp.predict(y, kernel=term)
    assert np.allclose(mu0, mu1)

    mu0 = gp.predict(y, t)
    mu1 = gp.predict(y, t, kernel=term)
    assert np.allclose(mu0, mu1)

    mu0 = gp.predict(y, t)
    mu1 = gp.predict(y, t, kernel=term1)
    mu2 = gp.predict(y, t, kernel=term2)
    assert np.allclose(mu0, mu1 + mu2)

    mu1 = gp.predict(y, kernel=term1)
    mu2 = term1.dot(x, np.zeros_like(x), gp.apply_inverse(y))
    assert np.allclose(mu1, mu2)


def test_errors():
    # Generate fake data
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.sort(np.random.uniform(-1, 12, 100))
    diag = np.random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)

    term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
    gp = celerite2.GaussianProcess(term)

    # Need to call compute first
    with pytest.raises(RuntimeError):
        gp.recompute()

    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)

    with pytest.raises(RuntimeError):
        gp.sample()

    # Sorted
    with pytest.raises(ValueError):
        gp.compute(x[::-1], diag=diag)

    # 1D
    with pytest.raises(ValueError):
        gp.compute(np.tile(x[:, None], (1, 5)), diag=diag)

    # Only one of diag and yerr
    with pytest.raises(ValueError):
        gp.compute(x, diag=diag, yerr=np.sqrt(diag))

    # Not positive definite
    # with pytest.raises(celerite2.driver.LinAlgError):
    with pytest.raises(Exception):
        gp.compute(x, diag=-10 * diag)

    # Not positive definite with `quiet`
    gp.compute(x, diag=-10 * diag, quiet=True)
    assert np.isinf(gp._log_det)
    assert gp._log_det < 0

    # Compute correctly
    gp.compute(x, diag=diag)
    gp.log_likelihood(y)

    # Dimension mismatch
    with pytest.raises(ValueError):
        gp.log_likelihood(y[:-1])

    with pytest.raises(ValueError):
        gp.log_likelihood(np.tile(y[:, None], (1, 5)))

    with pytest.raises(ValueError):
        gp.predict(y, t=np.tile(t[:, None], (1, 5)))
