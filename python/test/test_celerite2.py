# -*- coding: utf-8 -*-
import celerite as original_celerite
import numpy as np
import pytest
from celerite import terms as cterms

import celerite2
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


@pytest.mark.parametrize("oterm", test_terms)
def test_consistency(oterm):
    # Generate fake data
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    diag = np.random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)

    # Setup the original GP
    original_gp = original_celerite.GP(oterm)
    original_gp.compute(x, np.sqrt(diag))

    # Setup the new GP
    term = terms.OriginalCeleriteTerm(oterm)
    gp = celerite2.GaussianProcess(term)
    gp.compute(x, diag=diag)
    print(gp._W)
    print(gp._log_det)

    assert np.allclose(original_gp.log_likelihood(y), gp.log_likelihood(y))
