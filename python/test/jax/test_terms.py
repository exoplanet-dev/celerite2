# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import pytest

from celerite2 import terms as pyterms
from celerite2.testing import check_tensor_term

try:
    from jax.config import config
except ImportError:
    HAS_JAX = False
else:
    HAS_JAX = True

    config.update("jax_enable_x64", True)

    from celerite2.jax import terms

pytestmark = pytest.mark.skipif(not HAS_JAX, reason="jax is not installed")


def evaluate(x):
    assert x.dtype == np.float64 or x.dtype == np.int64
    return np.asarray(x)


compare_terms = partial(check_tensor_term, evaluate)


@pytest.mark.parametrize(
    "name,args",
    [
        ("RealTerm", dict(a=1.5, c=0.3)),
        ("ComplexTerm", dict(a=1.5, b=0.7, c=0.3, d=0.1)),
        ("SHOTerm", dict(S0=1.5, w0=2.456, Q=0.1)),
        (
            "SHOTerm",
            dict(S0=np.float64(1.5), w0=np.float64(2.456), Q=np.float64(3.4)),
        ),
        ("SHOTerm", dict(Sw4=1.5, w0=2.456, Q=3.4)),
        ("SHOTerm", dict(S_tot=1.5, w0=2.456, Q=3.4)),
        ("Matern32Term", dict(sigma=1.5, rho=3.5)),
        (
            "RotationTerm",
            dict(amp=1.5, Q0=2.1, deltaQ=0.5, period=1.3, mix=0.7),
        ),
    ],
)
def test_base_terms(name, args):
    term = getattr(terms, name)(**args)
    pyterm = getattr(pyterms, name)(**args)
    compare_terms(term, pyterm)

    compare_terms(terms.TermDiff(term), pyterms.TermDiff(pyterm))
    compare_terms(
        terms.IntegratedTerm(term, 0.5),
        pyterms.IntegratedTerm(pyterm, 0.5),
        atol=5e-6,
    )

    term0 = terms.SHOTerm(S0=1.0, w0=0.5, Q=1.5)
    pyterm0 = pyterms.SHOTerm(S0=1.0, w0=0.5, Q=1.5)
    compare_terms(term + term0, pyterm + pyterm0)
    compare_terms(term * term0, pyterm * pyterm0)

    term0 = terms.SHOTerm(S0=1.0, w0=0.5, Q=0.2)
    pyterm0 = pyterms.SHOTerm(S0=1.0, w0=0.5, Q=0.2)
    compare_terms(term + term0, pyterm + pyterm0)
    compare_terms(term * term0, pyterm * pyterm0)
