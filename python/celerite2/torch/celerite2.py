# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
import numpy as np
import torch
from torch import nn

from ..backprop import LinAlgError
from ..celerite2 import ConstantMean
from ..celerite2 import GaussianProcess as SuperGaussianProcess
from . import ops
from .terms import as_tensor


class GaussianProcess(nn.Module, SuperGaussianProcess):
    def __init__(self, kernel, t=None, *, mean=0.0, **kwargs):
        super().__init__()
        self._kernel = kernel
        if callable(mean):
            self._mean = mean
        else:
            self._mean = ConstantMean(mean)

        # Placeholders for storing data
        self._t = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        if t is not None:
            self.compute(t, **kwargs)

    def compute(
        self, t, *, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        # Check the input coordinates
        t = as_tensor(t)
        if check_sorted and torch.any(t[1:] - t[:-1] < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if t.ndim != 1:
            raise ValueError("the input coordinates must be one dimensional")

        # Save the diagonal
        self._t = as_tensor(t)
        self._diag = torch.empty_like(self._t)
        if yerr is None and diag is None:
            self._diag[:] = 0.0

        elif yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag[:] = as_tensor(yerr) ** 2

        else:
            self._diag[:] = as_tensor(diag)

        # Fill the celerite matrices
        (
            self._d,
            self._U,
            self._V,
            self._P,
        ) = self._kernel.get_celerite_matrices(self._t, self._diag)

        # Compute the Cholesky factorization
        try:
            self._d, self._W = ops.factor(self._d, self._U, self._V, self._P)
        except LinAlgError:
            if not quiet:
                raise
            self._log_det = -np.inf
            self._norm = np.inf
        else:
            self._log_det = torch.sum(torch.log(self._d))
            self._norm = -0.5 * (
                self._log_det + self._t.shape[0] * np.log(2 * np.pi)
            )

    def _process_input(self, y, *, require_vector=False):
        y = as_tensor(y)
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if self._t.shape[0] != y.shape[0]:
            raise ValueError("dimension mismatch")
        if require_vector and self._t.shape != y.shape:
            raise ValueError("'y' must be one dimensional")
        return y

    def apply_inverse(self, y):
        y = self._process_input(y)
        return ops.solve(self._U, self._P, self._d, self._W, y)

    def dot_tril(self, y):
        y = self._process_input(y)
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)

    def log_likelihood(self, y):
        y = self._process_input(y, require_vector=True)
        if not torch.isfinite(self._log_det):
            return as_tensor(-np.inf)
        loglike = self._norm - 0.5 * ops.norm(
            self._U, self._P, self._d, self._W, y - self._mean(self._t)
        )
        if not torch.isfinite(loglike):
            return -np.inf
        return loglike

    def forward(
        self, t, y, *, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        self.compute(
            t, yerr=yerr, diag=diag, check_sorted=check_sorted, quiet=quiet
        )
        return self.log_likelihood(y)

    def predict(
        self,
        y,
        t=None,
        *,
        return_cov=False,
        return_var=False,
        include_mean=True
    ):
        y = self._process_input(y, require_vector=True)

        alpha = ops.solve(
            self._U, self._P, self._d, self._W, y - self._mean(self._t)
        )

        if t is None:
            xs = self._t
            mu = y - self._diag * alpha

        else:
            xs = as_tensor(t)
            if xs.ndim != 1:
                raise ValueError("dimension mismatch")

            U_star, V_star, inds = self._kernel.get_conditional_mean_matrices(
                self._t, xs
            )
            mu = ops.conditional_mean(
                self._U, self._V, self._P, alpha, U_star, V_star, inds,
            )

            if include_mean:
                mu += self._mean(self._t)

        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        KxsT = self._kernel.get_value(xs[None, :] - self._t[:, None])
        if return_var:
            var = self._kernel.get_value(0.0) - torch.einsum(
                "ij,ij->j", KxsT, self.apply_inverse(KxsT)
            )
            return mu, var

        # Predictive covariance
        cov = self._kernel.get_value(xs[:, None] - xs[None, :])
        cov -= np.tensordot(KxsT, self.apply_inverse(KxsT), axes=(0, 0))
        return mu, cov

    def sample(self, *args, **kwargs):
        raise NotImplementedError("'sample' is not implemented in PyTorch")

    def sample_conditional(self, *args, **kwargs):
        raise NotImplementedError(
            "'sample_conditional' is not implemented in PyTorch"
        )
