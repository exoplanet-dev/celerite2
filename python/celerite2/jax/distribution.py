# -*- coding: utf-8 -*-

__all__ = ["CeleriteNormal"]
from jax import numpy as jnp
from jax import random as random

try:
    import numpyro  # noqa
except ImportError:

    class CeleriteNormal:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "pymc3 is required to use the CeleriteNormal distribution"
            )


else:
    from numpyro import distributions as dist

    class CeleriteNormal(dist.Distribution):
        support = dist.constraints.real_vector

        def __init__(self, gp, validate_args=None):
            self.gp = gp
            super().__init__(
                batch_shape=(),
                event_shape=jnp.shape(self.gp._t),
                validate_args=validate_args,
            )

        @dist.util.validate_sample
        def log_prob(self, value):
            return self.gp.log_likelihood(value)

        def sample(self, key, sample_shape=()):
            eps = random.normal(key, shape=self.event_shape + sample_shape)
            return jnp.moveaxis(self.gp.dot_tril(eps), 0, -1)
