# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
import numpy as np
from theano import tensor as tt

from ..ext import BaseGaussianProcess
from . import ops
from .distribution import CeleriteNormal

try:
    import pymc3 as pm
except ImportError:
    pm = None

CITATIONS = (
    ("celerite2:foremanmackey17", "celerite2:foremanmackey18"),
    r"""
@article{exoplanet:foremanmackey17,
   author = {{Foreman-Mackey}, D. and {Agol}, E. and {Ambikasaran}, S. and
             {Angus}, R.},
    title = "{Fast and Scalable Gaussian Process Modeling with Applications to
              Astronomical Time Series}",
  journal = {\aj},
     year = 2017,
    month = dec,
   volume = 154,
    pages = {220},
      doi = {10.3847/1538-3881/aa9332},
   adsurl = {http://adsabs.harvard.edu/abs/2017AJ....154..220F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
@article{exoplanet:foremanmackey18,
   author = {{Foreman-Mackey}, D.},
    title = "{Scalable Backpropagation for Gaussian Processes using Celerite}",
  journal = {Research Notes of the American Astronomical Society},
     year = 2018,
    month = feb,
   volume = 2,
   number = 1,
    pages = {31},
      doi = {10.3847/2515-5172/aaaf6c},
   adsurl = {http://adsabs.harvard.edu/abs/2018RNAAS...2a..31F},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
)


class GaussianProcess(BaseGaussianProcess):
    def as_tensor(self, tensor):
        return tt.as_tensor_variable(tensor).astype("float64")

    def zeros_like(self, tensor):
        return tt.zeros_like(tensor)

    def do_compute(self, quiet):
        if quiet:
            self._d, self._W, _ = ops.factor_quiet(
                self._a, self._U, self._V, self._P
            )
            self._log_det = tt.switch(
                tt.any(self._d < 0.0), -np.inf, tt.sum(tt.log(self._d))
            )

        else:
            self._d, self._W, _ = ops.factor(
                self._a, self._U, self._V, self._P
            )
            self._log_det = tt.sum(tt.log(self._d))

        self._norm = -0.5 * (
            self._log_det + self._t.shape[0] * np.log(2 * np.pi)
        )

    def check_sorted(self, t):
        return tt.opt.Assert()(t, tt.all(t[1:] - t[:-1] >= 0))

    def do_solve(self, y):
        return ops.solve(self._U, self._P, self._d, self._W, y)[0]

    def do_dot_tril(self, y):
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)[0]

    def do_norm(self, y):
        return ops.norm(self._U, self._P, self._d, self._W, y)[0]

    def do_conditional_mean(self, *args):
        return ops.conditional_mean(self._U, self._V, self._P, *args)

    def tensordot(self, a, b):
        return tt.tensordot(a, b, axes=(0, 0))

    def diagdot(self, a, b):
        return tt.batched_dot(a.T, b.T)

    def _add_citations_to_pymc3_model(self, **kwargs):
        if not pm:
            raise ImportError("pymc3 is required for the 'marginal' method")

        model = pm.modelcontext(kwargs.get("model", None))
        if not hasattr(model, "__citations__"):
            model.__citations__ = dict()
        model.__citations__["celerite2"] = CITATIONS

    def marginal(self, name, **kwargs):
        """Add the marginal likelihood to a PyMC3 model

        Args:
            name (str): The name of the random variable.
            observed (optional): The observed data

        Returns:
            A :class:`celerite2.theano.CeleriteNormal` distribution
            representing the marginal likelihood.
        """
        self._add_citations_to_pymc3_model(**kwargs)
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
            A :class:`pymc3.MvNormal` distribution representing the conditional
            density.
        """
        self._add_citations_to_pymc3_model(**kwargs)

        if t is None:
            shape = kwargs.pop("shape", len(y))
        else:
            shape = kwargs.pop("shape", len(t))

        mu, cov = self.predict(
            y,
            t=t,
            return_cov=True,
            include_mean=include_mean,
            kernel=kernel,
            _fast_mean=False,
        )
        return pm.MvNormal(name, mu=mu, cov=cov, shape=shape, **kwargs)
