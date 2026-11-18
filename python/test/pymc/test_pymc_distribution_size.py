import numpy as np
import pytest

pytest.importorskip("celerite2.pymc")

from celerite2.pymc import GaussianProcess, terms


@pytest.mark.parametrize("size", [(1,), (4,), (2, 3)])
def test_marginal_sample_size(size):
    pm = pytest.importorskip("pymc")
    t = np.linspace(0.0, 10.0, 8)
    kernel = terms.SHOTerm(sigma=1.0, rho=2.0, Q=1.5)

    with pm.Model():
        gp = GaussianProcess(kernel, t=t, diag=0.1)
        sample = pm.draw(gp.marginal("obs", size=size), random_seed=40582)

    with pm.Model():
        gp = GaussianProcess(kernel, t=t, diag=0.1)
        expected = pm.draw(
            gp.marginal("obs"), draws=int(np.prod(size)), random_seed=40582
        )

    assert sample.shape == size + (len(t),)
    np.testing.assert_array_equal(sample, expected.reshape(sample.shape))


def test_marginal_sample_size_posterior_predictive():
    pm = pytest.importorskip("pymc")
    t = np.linspace(0.0, 10.0, 8)
    with pm.Model():
        offset = pm.Normal("offset", 0.0, 2.0)
        gp = GaussianProcess(
            terms.SHOTerm(sigma=1.0, rho=2.0, Q=1.5),
            t=t,
            diag=0.1,
            mean=offset,
        )
        gp.marginal("obs", observed=np.sin(t))
        sample = pm.sample_posterior_predictive(
            [{"offset": np.array(0.0)}, {"offset": np.array(0.0)}],
            random_seed=40582,
            return_inferencedata=False,
            progressbar=False,
        )["obs"]
    assert sample.shape == (1, 2, len(t))
    assert np.all(np.isfinite(sample))
