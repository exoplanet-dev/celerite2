import numpy as np
import pytest

pytest.importorskip("celerite2.pymc")

try:
    from pymc.testing import assert_support_point_is_expected

    from celerite2.pymc import GaussianProcess, terms
    from celerite2.pymc.distribution import CeleriteNormalRV
except (ImportError, ModuleNotFoundError):
    pass


def test_celerite_normal_rv():
    # Test that ndims_params and ndim_supp have the expected value
    # now that they are created from signature
    celerite_normal = CeleriteNormalRV()
    assert celerite_normal.ndim_supp == 1
    assert tuple(celerite_normal.ndims_params) == (1, 0, 1, 1, 2, 2, 1)


@pytest.mark.parametrize(
    "t, mean, size, expected",
    [
        (np.arange(5, dtype=float), 0.0, None, np.full(5, 0.0)),
        (
            np.arange(5, dtype=float),
            np.arange(5, dtype=float),
            None,
            np.arange(5, dtype=float),
        ),
    ],
)
def test_celerite_normal_support_point(t, mean, size, expected):
    # Test that support point has the expected shape and value
    pm = pytest.importorskip("pymc")

    with pm.Model() as model:
        term = terms.SHOTerm(S0=1.0, w0=0.5, Q=3.0)
        gp = GaussianProcess(term, t=t, mean=mean)
        # NOTE: Name must be "x" for assert function to work
        gp.marginal("x", size=size)
    assert_support_point_is_expected(model, expected)


@pytest.mark.parametrize("mean", [0.0, 10.5, lambda t: 0.2 * (t - 5)])
def test_samples_include_mean(mean):
    pm = pytest.importorskip("pymc")
    t = np.linspace(0.0, 10.0, 8)
    kernel = terms.SHOTerm(sigma=1.0, rho=2.0, Q=1.5)

    with pm.Model():
        gp = GaussianProcess(kernel, t=t, diag=0.1, mean=mean)
        sample = pm.draw(gp.marginal("obs"), random_seed=40582)

    with pm.Model():
        zero_gp = GaussianProcess(kernel, t=t, diag=0.1)
        zero_sample = pm.draw(zero_gp.marginal("obs"), random_seed=40582)

    expected_mean = mean(t) if callable(mean) else mean
    np.testing.assert_allclose(
        sample, zero_sample + expected_mean, rtol=1e-12, atol=1e-12
    )
