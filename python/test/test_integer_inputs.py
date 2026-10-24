import numpy as np
import pytest

from celerite2 import terms


@pytest.mark.parametrize(
    "kernel",
    [
        terms.RealTerm(a=1.2, c=0.3),
        terms.ComplexTerm(a=1.2, b=0.1, c=0.3, d=0.4),
        terms.SHOTerm(S0=1.2, w0=0.3, Q=2.0),
        terms.TermConvolution(terms.RealTerm(a=1.2, c=0.3), delta=0.5),
    ],
)
@pytest.mark.parametrize("dtype", [np.int16, np.int64, np.float32])
@pytest.mark.parametrize("method", ["get_value", "get_psd"])
def test_numeric_evaluation_inputs(kernel, dtype, method):
    x = np.array([-2, 0, 1, 3], dtype=dtype)
    actual = getattr(kernel, method)(x)
    expected = getattr(kernel, method)(x.astype(float))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_minimum_integer_lag():
    kernel = terms.RealTerm(a=1.2, c=0.3)
    x = np.array([-32768], dtype=np.int16)
    np.testing.assert_array_equal(kernel.get_value(x), [0.0])


def test_integer_frequency_does_not_overflow_when_squared():
    kernel = terms.RealTerm(a=1.2, c=0.3)
    x = np.array([1000], dtype=np.int16)
    expected = np.sqrt(2 / np.pi) * 1.2 * 0.3 / (0.3**2 + 1000.0**2)
    np.testing.assert_allclose(kernel.get_psd(x), [expected], rtol=1e-14)


def test_extended_precision_input_is_preserved():
    kernel = terms.RealTerm(a=1.2, c=0.3)
    x = np.array([1.0], dtype=np.longdouble)
    assert kernel.get_value(x).dtype == np.dtype(np.longdouble)
    assert kernel.get_psd(x).dtype == np.dtype(np.longdouble)
