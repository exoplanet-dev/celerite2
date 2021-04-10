# -*- coding: utf-8 -*-
import numpy as np
import pytest
from jax.config import config

from celerite2 import terms

config.update("jax_enable_x64", True)

cterms = pytest.importorskip("celerite.terms")

test_terms = [
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
    + cterms.SHOTerm(
        log_S0=np.log(1.1), log_Q=np.log(0.15), log_omega0=np.log(1.2)
    ),
]


def _convert_kernel(celerite_kernel):
    if isinstance(celerite_kernel, cterms.TermSum):
        result = _convert_kernel(celerite_kernel.terms[0])
        for k in celerite_kernel.terms[1:]:
            result += _convert_kernel(k)
        return result
    elif isinstance(celerite_kernel, cterms.TermProduct):
        raise ValueError("TermProduct is not implemented")
    elif isinstance(celerite_kernel, cterms.RealTerm):
        raise ValueError("RealTerm is not implemented")
    elif isinstance(celerite_kernel, cterms.ComplexTerm):
        if not celerite_kernel.fit_b:
            return terms.CeleriteTerm(
                a=np.exp(celerite_kernel.log_a),
                b=0.0,
                c=np.exp(celerite_kernel.log_c),
                d=np.exp(celerite_kernel.log_d),
            )
        return terms.CeleriteTerm(
            a=np.exp(celerite_kernel.log_a),
            b=np.exp(2 * (celerite_kernel.log_b - celerite_kernel.log_a)),
            c=np.exp(celerite_kernel.log_c),
            d=np.exp(celerite_kernel.log_d),
        )
    elif isinstance(celerite_kernel, cterms.SHOTerm):
        return terms.SHOTerm(
            S0=np.exp(celerite_kernel.log_S0),
            Q=np.exp(celerite_kernel.log_Q),
            w0=np.exp(celerite_kernel.log_omega0),
        )
    elif isinstance(celerite_kernel, cterms.Matern32Term):
        return terms.Matern32Term(
            sigma=np.exp(celerite_kernel.log_sigma),
            rho=np.exp(celerite_kernel.log_rho),
        )
    raise NotImplementedError()


def _check_term(term, x, diag, K, psd_term):
    tau = x[:, None] - x[None, :]

    np.testing.assert_allclose(term.get_value(tau), K)

    # And the power spectrum
    omega = np.linspace(-10, 10, 500)
    np.testing.assert_allclose(psd_term.get_psd(omega), term.get_psd(omega))

    # Add in the diagonal
    K += np.diag(diag)

    # Matrix vector multiply
    y = np.sin(x)
    value = term.dot(x, diag, y)
    np.testing.assert_allclose(y, np.sin(x))
    np.testing.assert_allclose(value, K @ y)

    # Matrix-matrix multiply
    y = np.vstack([x]).T
    value = term.dot(x, diag, y)
    np.testing.assert_allclose(value, K @ y)


@pytest.mark.parametrize("oterm", test_terms)
def test_consistency(oterm):
    # Check that the coefficients are all correct
    term = _convert_kernel(oterm)

    # Make sure that the covariance matrix is right
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    diag = np.random.uniform(0.1, 0.3, len(x))
    np.testing.assert_allclose(oterm.get_value(x), term.get_value(x))
    _check_term(
        term,
        x,
        diag,
        oterm.get_value(x[:, None] - x[None, :]),
        oterm,
    )


def test_matern32():
    sigma = 3.5
    rho = 2.75
    term = terms.Matern32Term(sigma=sigma, rho=rho)

    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    diag = np.random.uniform(0.1, 0.3, len(x))
    tau = np.abs(x[:, None] - x[None, :])
    f = np.sqrt(3) * tau / rho
    K = sigma ** 2 * (1 + f) * np.exp(-f)

    _check_term(
        term,
        x,
        diag,
        K,
        (
            terms.SHOTerm(
                sigma=sigma, rho=2 * np.pi * rho / np.sqrt(3), Q=0.5 + 1e-8
            )
        ),
    )
