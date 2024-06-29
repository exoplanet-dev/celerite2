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
