# -*- coding: utf-8 -*-
import celerite2
import numpy as np
import pytest
from celerite2 import terms as pyterms
from celerite2.testing import check_gp_models
from celerite2.theano import GaussianProcess, terms
from celerite2.theano.celerite2 import CITATIONS

term_mark = pytest.mark.parametrize(
    "name,args",
    [
        ("RealTerm", dict(a=1.5, c=0.3)),
        ("ComplexTerm", dict(a=1.5, b=0.7, c=0.3, d=0.1)),
        ("SHOTerm", dict(S0=1.5, w0=2.456, Q=0.1)),
        ("SHOTerm", dict(S0=1.5, w0=2.456, Q=3.4)),
        ("SHOTerm", dict(sigma=1.5, w0=2.456, Q=3.4)),
        ("Matern32Term", dict(sigma=1.5, rho=3.5)),
        ("RotationTerm", dict(sigma=1.5, Q0=2.1, dQ=0.5, period=1.3, f=0.7)),
    ],
)


@pytest.fixture
def data():
    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.sort(np.random.uniform(-1, 12, 100))
    diag = np.random.uniform(0.1, 0.3, len(x))
    y = np.sin(x)
    return x, diag, y, t


@term_mark
@pytest.mark.parametrize("mean", [0.0, 10.5])
def test_consistency(name, args, mean, data):
    x, diag, y, t = data

    term = getattr(terms, name)(**args)
    gp = GaussianProcess(term, mean=mean)
    gp.compute(x, diag=diag)

    pyterm = getattr(pyterms, name)(**args)
    pygp = celerite2.GaussianProcess(pyterm, mean=mean)
    pygp.compute(x, diag=diag)

    check_gp_models(lambda x: x.eval(), gp, pygp, y, t)


def test_errors(data):
    x, diag, y, t = data

    term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
    gp = GaussianProcess(term)

    # Need to call compute first
    with pytest.raises(RuntimeError):
        gp.log_likelihood(y)

    # Sorted
    with pytest.raises(AssertionError):
        gp.compute(x[::-1], diag=diag)
        gp._d.eval()

    # 1D
    with pytest.raises(ValueError):
        gp.compute(np.tile(x[:, None], (1, 5)), diag=diag)

    # Only one of diag and yerr
    with pytest.raises(ValueError):
        gp.compute(x, diag=diag, yerr=np.sqrt(diag))

    # Not positive definite
    with pytest.raises(celerite2.backprop.LinAlgError):
        gp.compute(x, diag=-10 * diag)
        gp._d.eval()

    # Not positive definite with `quiet`
    gp.compute(x, diag=-10 * diag, quiet=True)
    ld = gp._log_det.eval()
    assert np.isinf(ld)
    assert ld < 0

    # Compute correctly
    gp.compute(x, diag=diag)
    gp.log_likelihood(y).eval()

    # Dimension mismatch
    with pytest.raises(ValueError):
        gp.log_likelihood(y[:-1]).eval()

    with pytest.raises(ValueError):
        gp.log_likelihood(np.tile(y[:, None], (1, 5)))

    with pytest.raises(ValueError):
        gp.predict(y, t=np.tile(t[:, None], (1, 5)))


def test_marginal(data):
    import pymc3 as pm

    x, diag, y, t = data

    with pm.Model() as model:
        term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
        gp = GaussianProcess(term, t=x, diag=diag)
        gp.marginal("obs", observed=y)

        assert np.allclose(
            model.fastfn(model.logpt)(model.test_point),
            model.fastfn(gp.log_likelihood(y))(model.test_point),
        )


def test_citations(data):
    import pymc3 as pm

    x, diag, y, t = data

    with pm.Model() as model:
        term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
        gp = GaussianProcess(term, t=x, diag=diag)
        gp.marginal("obs", observed=y)
        assert model.__citations__["celerite2"] == CITATIONS
