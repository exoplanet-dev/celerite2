import numpy as np
import pytest

pytest.importorskip("celerite2.jax")
pytest.importorskip("numpyro")

from jax import jit, random

from celerite2.jax import GaussianProcess, terms


@pytest.mark.parametrize("mean", [0.0, 10.5, lambda t: 0.2 * (t - 5)])
@pytest.mark.parametrize("sample_shape", [(), (4,)])
def test_samples_include_mean(mean, sample_shape):
    t = np.linspace(0.0, 10.0, 8)
    kernel = terms.SHOTerm(sigma=1.0, rho=2.0, Q=1.5)
    gp = GaussianProcess(kernel, t=t, diag=0.1, mean=mean)
    zero_gp = GaussianProcess(kernel, t=t, diag=0.1)
    key = random.PRNGKey(40582)

    sample = jit(lambda k: gp.numpyro_dist().sample(k, sample_shape))(key)
    zero_sample = zero_gp.numpyro_dist().sample(key, sample_shape)
    expected_mean = mean(t) if callable(mean) else mean
    np.testing.assert_allclose(
        sample, zero_sample + expected_mean, rtol=1e-12, atol=1e-12
    )
