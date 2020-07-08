# -*- coding: utf-8 -*-
import numpy as np
import pytest
from celerite import terms as oterms

from celerite2 import terms

term_pairs = [
    (
        oterms.RealTerm(log_a=np.log(1.5), log_c=np.log(1.1)),
        terms.RealTerm(a=1.5, c=1.1),
    ),
    (
        oterms.RealTerm(log_a=np.log(12.345), log_c=np.log(1.5))
        + oterms.RealTerm(log_a=np.log(0.5), log_c=np.log(1.1234)),
        terms.RealTerm(a=12.345, c=1.5) + terms.RealTerm(a=0.5, c=1.1234),
    ),
    (
        oterms.ComplexTerm(
            log_a=np.log(10.0), log_c=np.log(5.6), log_d=np.log(2.1)
        ),
        terms.ComplexTerm(a=10.0, b=0.0, c=5.6, d=2.1),
    ),
    (
        oterms.ComplexTerm(
            log_a=np.log(7.435),
            log_b=np.log(0.5),
            log_c=np.log(1.102),
            log_d=np.log(1.05),
        ),
        terms.ComplexTerm(a=7.435, b=0.5, c=1.102, d=1.05),
    ),
    (
        oterms.SHOTerm(
            log_S0=np.log(1.1), log_Q=np.log(0.1), log_omega0=np.log(1.2)
        ),
        terms.SHOTerm(S0=1.1, Q=0.1, w0=1.2),
    ),
    (
        oterms.SHOTerm(
            log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
        ),
        terms.SHOTerm(S0=1.1, Q=2.5, w0=1.2),
    ),
    (
        oterms.SHOTerm(
            log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
        )
        + oterms.RealTerm(log_a=np.log(1.345), log_c=np.log(2.4)),
        terms.SHOTerm(S0=1.1, Q=2.5, w0=1.2) + terms.RealTerm(a=1.345, c=2.4),
    ),
    (
        oterms.SHOTerm(
            log_S0=np.log(1.1), log_Q=np.log(2.5), log_omega0=np.log(1.2)
        )
        * oterms.RealTerm(log_a=np.log(1.345), log_c=np.log(2.4)),
        terms.SHOTerm(S0=1.1, Q=2.5, w0=1.2) * terms.RealTerm(a=1.345, c=2.4),
    ),
]


@pytest.mark.parametrize("oterm, term", term_pairs)
def test_consistency(oterm, term):
    for v1, v2 in zip(oterm.get_all_coefficients(), term.get_coefficients()):
        assert np.allclose(v1, v2)
    for v1, v2 in zip(
        terms.OriginalCeleriteTerm(oterm).get_coefficients(),
        term.get_coefficients(),
    ):
        assert np.allclose(v1, v2)

    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    diag = np.random.uniform(0.1, 0.3, len(x))
    assert np.allclose(oterm.get_value(x), term.get_value(x))

    tau = x[:, None] - x[None, :]
    K = term.get_value(tau)
    assert np.allclose(oterm.get_value(tau), K)

    omega = np.linspace(-10, 10, 500)
    assert np.allclose(oterm.get_psd(omega), term.get_psd(omega))

    y = np.sin(x)
    K[np.diag_indices_from(K)] += diag
    value = term.dot(x, diag, y)

    assert np.allclose(y, np.sin(x))
    assert np.allclose(value, np.dot(K, y))
