# -*- coding: utf-8 -*-

__all__ = [
    "factor",
    "solve",
    "norm",
    "dot_tril",
    "matmul",
    "conditional_mean",
    "searchsorted",
]
from functools import wraps

import numpy as np
import torch
from torch.autograd import Function

from .. import driver, ext


def wrap_forward(func):
    @wraps(func)
    def wrapped(ctx, *inputs):
        in_tensors = tuple(x.detach() for x in inputs)
        in_arrays = tuple(x.numpy() for x in in_tensors)
        out_arrays = func(*in_arrays)
        out_tensors = tuple(
            torch.as_tensor(x, dtype=torch.float64) for x in out_arrays
        )
        ctx.save_for_backward(*in_tensors, *out_tensors)
        return out_tensors

    return wrapped


def wrap_backward(func):
    @wraps(func)
    def wrapped(ctx, *grads):
        grad_tensors = tuple(x.detach() for x in grads)
        grad_arrays = tuple(x.numpy() for x in grad_tensors)
        saved_arrays = tuple(x.detach().numpy() for x in ctx.saved_tensors)
        out_arrays = func(*saved_arrays, *grad_arrays)
        out_tensors = tuple(
            torch.as_tensor(x, dtype=torch.float64) for x in out_arrays
        )
        return out_tensors

    return wrapped


class Factor(Function):
    @staticmethod
    def forward(ctx, a, U, V, P):
        return wrap_forward(ext.factor_fwd)(ctx, a, U, V, P)[:2]

    @staticmethod
    def backward(ctx, bd, bW):
        return wrap_backward(ext.factor_rev)(ctx, bd, bW)


class Solve(Function):
    @staticmethod
    def forward(ctx, U, P, d, W, Y):
        return wrap_forward(ext.solve_fwd)(ctx, U, P, d, W, Y)[0]

    @staticmethod
    def backward(ctx, bX):
        return wrap_backward(ext.solve_rev)(ctx, bX)


class Norm(Function):
    @staticmethod
    def forward(ctx, U, P, d, W, Y):
        return wrap_forward(ext.norm_fwd)(ctx, U, P, d, W, Y)[0]

    @staticmethod
    def backward(ctx, bX):
        return wrap_backward(ext.norm_rev)(ctx, bX)


class DotTril(Function):
    @staticmethod
    def forward(ctx, U, P, d, W, Y):
        return wrap_forward(ext.dot_tril_fwd)(ctx, U, P, d, W, Y)[0]

    @staticmethod
    def backward(ctx, bX):
        return wrap_backward(ext.dot_tril_rev)(ctx, bX)


class Matmul(Function):
    @staticmethod
    def forward(ctx, a, U, V, P, Y):
        return wrap_forward(ext.matmul_fwd)(ctx, a, U, V, P, Y)[0]

    @staticmethod
    def backward(ctx, bX):
        return wrap_backward(ext.matmul_rev)(ctx, bX)


class ConditionalMean(Function):
    @staticmethod
    def forward(ctx, U, V, P, alpha, U_star, V_star, inds):
        inds_ = inds.detach().numpy()
        mu_ = np.empty(inds_.shape, dtype=np.float64)
        mu_ = driver.conditional_mean(
            U.detach().numpy(),
            V.detach().numpy(),
            P.detach().numpy(),
            alpha.detach().numpy(),
            U_star.detach().numpy(),
            V_star.detach().numpy(),
            inds_,
            mu_,
        )
        return torch.as_tensor(mu_, dtype=torch.double)


class Searchsorted(Function):
    @staticmethod
    def forward(ctx, x, t):
        ctx.save_for_backward(x, t)
        x_ = x.detach().numpy()
        t_ = t.detach().numpy()
        inds = np.searchsorted(x_, t_)
        return torch.as_tensor(inds, dtype=torch.int64)

    @staticmethod
    def backward(ctx, grad):
        x, t = ctx.saved_tensors
        return torch.zeros_like(x), torch.zeros_like(t)


def factor(*args):
    return Factor.apply(*args)


def solve(*args):
    return Solve.apply(*args)


def norm(*args):
    return Norm.apply(*args)


def dot_tril(*args):
    return DotTril.apply(*args)


def matmul(*args):
    return Matmul.apply(*args)


def conditional_mean(*args):
    return ConditionalMean.apply(*args)


def searchsorted(*args):
    return Searchsorted.apply(*args)
