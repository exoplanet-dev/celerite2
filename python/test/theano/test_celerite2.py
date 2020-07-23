# -*- coding: utf-8 -*-
import numpy as np
import pytest

import celerite2
from celerite2 import terms as pyterms

try:
    import theano
except ImportError:
    HAS_THEANO = False
else:
    from celerite2.theano import GaussianProcess, terms

    HAS_THEANO = True


pytestmark = pytest.mark.skipif(
    not HAS_THEANO, reason="Theano is not installed"
)


term_mark = pytest.mark.parametrize(
    "name,args",
    [
        ("RealTerm", dict(a=1.5, c=0.3)),
        ("ComplexTerm", dict(a=1.5, b=0.7, c=0.3, d=0.1)),
        ("SHOTerm", dict(S0=1.5, w0=2.456, Q=0.1)),
        ("SHOTerm", dict(S0=1.5, w0=2.456, Q=3.4)),
        ("SHOTerm", dict(Sw4=1.5, w0=2.456, Q=3.4)),
        ("SHOTerm", dict(S_tot=1.5, w0=2.456, Q=3.4)),
        ("Matern32Term", dict(sigma=1.5, rho=3.5)),
        (
            "RotationTerm",
            dict(amp=1.5, Q0=2.1, deltaQ=0.5, period=1.3, mix=0.7),
        ),
    ],
)


@term_mark
def test_consistency(name, args):
    # Generate fake data
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.sort(np.random.uniform(-1, 12, 100))
    diag = np.random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)

    term = getattr(terms, name)(**args)
    gp = GaussianProcess(term)
    gp.compute(x, diag=diag)

    pyterm = getattr(pyterms, name)(**args)
    pygp = celerite2.GaussianProcess(pyterm)
    pygp.compute(x, diag=diag)

    # "log_likelihood" method
    assert np.allclose(pygp.log_likelihood(y), gp.log_likelihood(y).eval())

    # "predict" method
    for args in [
        dict(return_cov=False, return_var=False),
        dict(return_cov=False, return_var=True),
        dict(return_cov=True, return_var=False),
    ]:
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                pygp.predict(y, **args),
                theano.function([], gp.predict(y, **args))(),
            )
        )
        assert all(
            np.allclose(a, b)
            for a, b in zip(
                pygp.predict(y, t=t, **args),
                theano.function([], gp.predict(y, t=t, **args))(),
            )
        )

    assert np.allclose(pygp.dot_tril(y), gp.dot_tril(y).eval())
