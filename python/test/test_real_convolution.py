import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve

from celerite2 import GaussianProcess, terms


def _averaged_kernel(lag, delta, components):
    def integrand(u):
        value = sum(
            a * np.exp(-c * abs(lag + delta * u)) for a, c in components
        )
        return (1 - abs(u)) * value

    points = [0.0]
    if abs(lag) < delta:
        points.append(-lag / delta)
    return quad(integrand, -1, 1, points=points, epsabs=1e-13)[0]


@pytest.mark.parametrize("delta", [0.1, 1e-3, 1e-5, 1e-7, 1e-9])
@pytest.mark.parametrize(
    "components", [[(1.2, 0.08)], [(1.2, 0.08), (0.5, 0.2)]]
)
def test_short_exposure_real_kernel(delta, components):
    kernel = terms.RealTerm(a=components[0][0], c=components[0][1])
    for a, c in components[1:]:
        kernel += terms.RealTerm(a=a, c=c)
    kernel = terms.TermConvolution(kernel, delta=delta)
    lags = np.array([0.0, 0.25 * delta, 0.5 * delta, delta, 1.0])
    expected = [_averaged_kernel(lag, delta, components) for lag in lags]
    np.testing.assert_allclose(
        kernel.get_value(lags), expected, rtol=2e-12, atol=2e-12
    )


@pytest.mark.parametrize("delta", [0.1, 1e-3, 1e-5, 1e-7, 1e-9])
def test_short_exposure_real_factorization(delta):
    components = [(1.2, 0.08)]
    kernel = terms.TermConvolution(terms.RealTerm(a=1.2, c=0.08), delta=delta)
    t = np.arange(8.0)
    y = np.sin(t)
    values = np.array([_averaged_kernel(lag, delta, components) for lag in t])
    expected_cov = values[
        np.abs(t[:, None] - t[None, :]).astype(int)
    ] + 0.04 * np.eye(8)
    expected_inverse = cho_solve(cho_factor(expected_cov), y)
    gp = GaussianProcess(kernel)
    gp.compute(t, diag=0.04)
    np.testing.assert_allclose(
        kernel.to_dense(t, np.full(8, 0.04)),
        expected_cov,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        gp.apply_inverse(y), expected_inverse, rtol=2e-11, atol=2e-11
    )
