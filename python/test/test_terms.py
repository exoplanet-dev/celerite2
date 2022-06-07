# -*- coding: utf-8 -*-
import numpy as np
import pytest
from celerite import terms as cterms

from celerite2 import terms

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


def _convert_kernel(celerite_kernel):
    if isinstance(celerite_kernel, cterms.TermSum):
        result = _convert_kernel(celerite_kernel.terms[0])
        for k in celerite_kernel.terms[1:]:
            result += _convert_kernel(k)
        return result
    elif isinstance(celerite_kernel, cterms.TermProduct):
        return _convert_kernel(celerite_kernel.k1) * _convert_kernel(
            celerite_kernel.k2
        )
    elif isinstance(celerite_kernel, cterms.RealTerm):
        return terms.RealTerm(
            a=np.exp(celerite_kernel.log_a), c=np.exp(celerite_kernel.log_c)
        )
    elif isinstance(celerite_kernel, cterms.ComplexTerm):
        if not celerite_kernel.fit_b:
            return terms.ComplexTerm(
                a=np.exp(celerite_kernel.log_a),
                b=0.0,
                c=np.exp(celerite_kernel.log_c),
                d=np.exp(celerite_kernel.log_d),
            )
        return terms.ComplexTerm(
            a=np.exp(celerite_kernel.log_a),
            b=np.exp(celerite_kernel.log_b),
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


@pytest.mark.parametrize("oterm", test_terms)
def test_consistency(oterm):
    # Check that the coefficients are all correct
    term = _convert_kernel(oterm)
    for v1, v2 in zip(oterm.get_all_coefficients(), term.get_coefficients()):
        assert np.allclose(v1, v2)
    for v1, v2 in zip(
        terms.OriginalCeleriteTerm(oterm).get_coefficients(),
        term.get_coefficients(),
    ):
        assert np.allclose(v1, v2)

    # Make sure that the covariance matrix is right
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    diag = np.random.uniform(0.1, 0.3, len(x))
    assert np.allclose(oterm.get_value(x), term.get_value(x))

    tau = x[:, None] - x[None, :]
    K = term.get_value(tau)
    assert np.allclose(oterm.get_value(tau), K)

    # And the power spectrum
    omega = np.linspace(-10, 10, 500)
    assert np.allclose(oterm.get_psd(omega), term.get_psd(omega))

    # Add in the diagonal
    K[np.diag_indices_from(K)] += diag

    # Matrix vector multiply
    y = np.sin(x)
    value = term.dot(x, diag, y)
    assert np.allclose(y, np.sin(x))
    assert np.allclose(value, np.dot(K, y))

    # Matrix-matrix multiply
    y = np.vstack([x]).T
    value = term.dot(x, diag, y)
    assert np.allclose(value, np.dot(K, y))
