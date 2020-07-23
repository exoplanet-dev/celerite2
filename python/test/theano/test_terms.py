# -*- coding: utf-8 -*-
import numpy as np
import pytest

from celerite2 import terms as pyterms

try:
    from celerite2.theano import terms
except ImportError:
    HAS_THEANO = False
else:
    HAS_THEANO = True


pytestmark = pytest.mark.skipif(
    not HAS_THEANO, reason="Theano is not installed"
)


def compare(tensor, array, text):
    value = tensor.eval()
    if array.size == 0:
        assert value.size == 0
        return
    resid = np.abs(array - value)
    coords = np.unravel_index(np.argmax(resid), array.shape)
    assert np.allclose(
        value, array
    ), f"resid: {resid.max()}; coords: {coords}; message: {text}"


def compare_terms(term, pyterm):
    tensors = term.coefficients
    arrays = pyterm.get_coefficients()
    inds = np.argsort(tensors[0].eval())
    pyinds = np.argsort(arrays[0])
    for n, (tensor, array) in enumerate(zip(tensors[:2], arrays[:2])):
        compare(tensor[inds], array[pyinds], f"real coefficients {n}")

    inds = np.argsort(tensors[2].eval())
    pyinds = np.argsort(arrays[2])
    for n, (tensor, array) in enumerate(zip(tensors[2:], arrays[2:])):
        compare(tensor[inds], array[pyinds], f"complex coefficients {n}")

    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.random.uniform(-1, 11, 75)
    diag = np.random.uniform(0.1, 0.3, len(x))

    # This is a hack to deal with the fact that the theano interface doesn't
    # always propduce matrices with the same column order
    tensors = term.get_celerite_matrices(x, diag)
    arrays = pyterm.get_celerite_matrices(x, diag)
    inds = np.argsort(tensors[1].eval()[0, :])
    pyinds = np.argsort(arrays[1][0, :])
    for n, (tensor, array) in enumerate(zip(tensors, arrays)):
        if n >= 1:
            compare(tensor[:, inds], array[:, pyinds], f"matrix {n}")
        else:
            compare(tensor, array, f"matrix {n}")

    # Same hack again...
    tensors = term.get_conditional_mean_matrices(x, t)
    arrays = pyterm.get_conditional_mean_matrices(x, t)
    compare(tensors[-1], arrays[-1], "sorted inds")
    inds = np.argsort(tensors[0].eval()[0, :])
    pyinds = np.argsort(arrays[0][0, :])
    for n, (tensor, array) in enumerate(zip(tensors[:2], arrays[:2])):
        compare(tensor[:, inds], array[:, pyinds], f"conditional matrix {n}")

    compare(term.to_dense(x, diag), pyterm.to_dense(x, diag), "to_dense")

    tau = x[:, None] - x[None, :]
    compare(term.get_value(tau), pyterm.get_value(tau), "get_value")

    omega = np.linspace(-10, 10, 500)
    compare(term.get_psd(omega), pyterm.get_psd(omega), "get_psd")

    y = np.reshape(np.sin(x), (len(x), 1))
    compare(term.dot(x, diag, y), pyterm.dot(x, diag, y), "dot")


@pytest.mark.parametrize(
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
def test_base_terms(name, args):
    term = getattr(terms, name)(**args)
    pyterm = getattr(pyterms, name)(**args)
    compare_terms(term, pyterm)

    compare_terms(terms.TermDiff(term), pyterms.TermDiff(pyterm))
    compare_terms(
        terms.IntegratedTerm(term, 0.5), pyterms.IntegratedTerm(pyterm, 0.5)
    )

    term0 = terms.SHOTerm(S0=1.0, w0=0.5, Q=1.5)
    pyterm0 = pyterms.SHOTerm(S0=1.0, w0=0.5, Q=1.5)
    compare_terms(term + term0, pyterm + pyterm0)
    compare_terms(term * term0, pyterm * pyterm0)

    term0 = terms.SHOTerm(S0=1.0, w0=0.5, Q=0.2)
    pyterm0 = pyterms.SHOTerm(S0=1.0, w0=0.5, Q=0.2)
    compare_terms(term + term0, pyterm + pyterm0)
    compare_terms(term * term0, pyterm * pyterm0)
