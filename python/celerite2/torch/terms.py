# -*- coding: utf-8 -*-

__all__ = ["Term"]
import torch
from torch import nn as nn


class Term(nn.Module):
    def __add__(self, b):
        return TermSum(self, b)

    def __radd__(self, b):
        return TermSum(b, self)

    def __mul__(self, b):
        return TermProduct(self, b)

    def __rmul__(self, b):
        return TermProduct(b, self)

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_celerite_matrices(self, x, diag):
        x = torch.as_tensor(x)
        diag = torch.as_tensor(diag)

        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        a = diag + torch.sum(ar) + torch.sum(ac)

        arg = dc[None, :] * x[:, None]
        cos = torch.cos(arg)
        sin = torch.sin(arg)
        z = torch.zeros_like(x)

        U = torch.cat(
            (
                ar[None, :] + z[:, None],
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )

        V = torch.cat(
            (torch.ones_like(ar)[None, :] + z[:, None], cos, sin), axis=1,
        )

        dx = x[1:] - x[:-1]
        c = torch.cat((cr, cc, cc))
        P = torch.exp(-c[None, :] * dx[:, None])

        return a, U, V, P

    def forward(self, x, diag):
        return self.get_celerite_matrices(x, diag)


class RealTerm(Term):
    def __init__(self, *, a, c):
        self.a = torch.as_tensor(a)
        self.c = torch.as_tensor(c)

    def get_coefficients(self):
        e = torch.empty(0)
        if self.a.ndim:
            return self.a, self.c, e, e, e, e
        return torch.tensor([self.a]), torch.tensor([self.c]), e, e, e, e


class ComplexTerm(Term):
    def __init__(self, *, a, b, c, d):
        self.a = torch.as_tensor(a)
        self.b = torch.as_tensor(b)
        self.c = torch.as_tensor(c)
        self.d = torch.as_tensor(d)

    def get_coefficients(self):
        e = torch.empty(0)
        if self.a.ndim:
            return (
                e,
                e,
                self.a,
                self.b,
                self.c,
                self.d,
            )
        return (
            e,
            e,
            torch.tensor([self.a]),
            torch.tensor([self.b]),
            torch.tensor([self.c]),
            torch.tensor([self.d]),
        )
