# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
import numpy as np
import torch
from torch import nn

from ..backprop import LinAlgError
from ..ext import BaseGaussianProcess
from . import ops
from .terms import as_tensor


class GaussianProcess(nn.Module, BaseGaussianProcess):
    def __init__(self, kernel, t=None, *, mean=0.0, **kwargs):
        super().__init__()
        self.kernel = kernel
        self.mean = mean

        # Placeholders for storing data
        self._t = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        if t is not None:
            self.compute(t, **kwargs)

    def as_tensor(self, tensor):
        return as_tensor(tensor)

    def zeros_like(self, tensor):
        return torch.zeros_like(tensor)

    def do_compute(self, quiet):
        # Compute the Cholesky factorization
        try:
            self._d, self._W = ops.factor(self._a, self._U, self._V, self._P)
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

    def check_sorted(self, t):
        if torch.any(t[1:] - t[:-1] < 0.0):
            raise ValueError("the input coordinates must be sorted")
        return t

    def do_solve(self, y):
        return ops.solve(self._U, self._P, self._d, self._W, y)

    def do_dot_tril(self, y):
        return ops.dot_tril(self._U, self._P, self._d, self._W, y)

    def do_norm(self, y):
        return ops.norm(self._U, self._P, self._d, self._W, y)

    def do_conditional_mean(self, *args):
        return ops.conditional_mean(self._U, self._V, self._P, *args)

    def tensordot(self, a, b):
        return torch.einsum("ij,ik->jk", a, b)

    def diagdot(self, a, b):
        return torch.einsum("ij,ij->j", a, b)
