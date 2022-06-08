# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess", "ConditionalDistribution"]

import numpy as np
import theano.tensor as tt

from celerite2.citation import CITATIONS
from celerite2.core import BaseConditionalDistribution, BaseGaussianProcess
from celerite2.pymc3 import ops


class ConditionalDistribution(BaseConditionalDistribution):
    def _do_general_matmul(self, c, U1, V1, U2, V2, inp, target):
        target += ops.general_matmul_lower(
            self._xs, self.gp._t, c, U2, V1, inp
        )[0]
        target += ops.general_matmul_upper(
            self._xs, self.gp._t, c, V2, U1, inp
        )[0]
        return target

    def _diagdot(self, a, b):
        return tt.batched_dot(a.T, b.T)


class GaussianProcess(BaseGaussianProcess):
    conditional_distribution = ConditionalDistribution

    def _as_tensor(self, tensor):
        return tt.as_tensor_variable(tensor).astype("float64")

    def _zeros_like(self, tensor):
        return tt.zeros_like(tensor)

    def _do_compute(self, quiet):
        if quiet:
            self._d, self._W, _ = ops.factor_quiet(
                self._t, self._c, self._a, self._U, self._V
            )
            self._log_det = tt.switch(
                tt.any(self._d <= 0.0), -np.inf, tt.sum(tt.log(self._d))
            )

        else:
            self._d, self._W, _ = ops.factor(
                self._t, self._c, self._a, self._U, self._V
            )
            self._log_det = tt.sum(tt.log(self._d))

        self._norm = -0.5 * (self._log_det + self._size * np.log(2 * np.pi))

    def _check_sorted(self, t):
        return tt.opt.Assert()(t, tt.all(t[1:] - t[:-1] >= 0))

    def _do_solve(self, y):
        z = ops.solve_lower(self._t, self._c, self._U, self._W, y)[0]
        z /= self._d[:, None]
        z = ops.solve_upper(self._t, self._c, self._U, self._W, z)[0]
        return z

    def _do_dot_tril(self, y):
        z = y * np.sqrt(self._d)[:, None]
        z += ops.matmul_lower(self._t, self._c, self._U, self._W, z)[0]
        return z

    def _do_norm(self, y):
        alpha = ops.solve_lower(
            self._t, self._c, self._U, self._W, y[:, None]
        )[0][:, 0]
        return tt.sum(alpha**2 / self._d)

    def _add_citations_to_pymc_model(self, **kwargs):
        import pymc3 as pm

        model = pm.modelcontext(kwargs.get("model", None))
        if not hasattr(model, "__citations__"):
            model.__citations__ = dict()
        model.__citations__["celerite2"] = CITATIONS

    def marginal(self, name, **kwargs):
        """Add the marginal likelihood to a PyMC model

        Args:
            name (str): The name of the random variable.
            observed (optional): The observed data

        Returns:
            A :class:`celerite2.pymc3.CeleriteNormal` distribution
            representing the marginal likelihood.
        """
        from celerite2.pymc3.distribution import CeleriteNormal

        self._add_citations_to_pymc_model(**kwargs)
        return CeleriteNormal(name, self, **kwargs)

    def conditional(
        self, name, y, t=None, include_mean=True, kernel=None, **kwargs
    ):
        """Add a variable representing the conditional density to a PyMC3 model

        .. note:: The performance of this method will generally be poor since
            the sampler will numerically sample this parameter. Depending on
            your use case, you might be better served by tracking the results
            of :func:`GaussianProcess.predict` using ``Deterministic``
            variables and computing the predictions as a postprocessing step.

        Args:
            name (str): The name of the random variable
            y (shape[N]): The observations at coordinates ``x`` from
                :func:`GausianProcess.compute`.
            t (shape[M], optional): The independent coordinates where the
                prediction should be made. If this is omitted the coordinates
                will be assumed to be ``x`` from
                :func:`GaussianProcess.compute` and an efficient method will
                be used to compute the mean prediction.
            include_mean (bool, optional): Include the mean function in the
                prediction.
            kernel (optional): If provided, compute the conditional
                distribution using a different kernel. This is generally used
                to separate the contributions from different model components.

        Returns:
            A :class:`pm.MvNormal` distribution representing the conditional
            density.
        """
        import pymc3 as pm

        self._add_citations_to_pymc_model(**kwargs)

        if t is None:
            shape = kwargs.pop("shape", len(y))
        else:
            shape = kwargs.pop("shape", len(t))

        cond = self.condition(
            y,
            t=t,
            include_mean=include_mean,
            kernel=kernel,
        )
        return pm.MvNormal(
            name, mu=cond.mean, cov=cond.covariance, shape=shape, **kwargs
        )
