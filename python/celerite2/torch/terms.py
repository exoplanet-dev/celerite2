# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
    "TermProduct",
    "TermDiff",
    "IntegratedTerm",
    "RealTerm",
    "ComplexTerm",
    "SHOTerm",
    "Matern32Term",
    "RotationTerm",
]
from functools import partial

import numpy as np
import torch
from torch import nn

from . import ops

as_tensor = partial(torch.as_tensor, dtype=torch.double)


class Term(nn.Module):
    def forward(self, x, diag):
        return self.get_celerite_matrices(x, diag)

    def __add__(self, b):
        return TermSum(self, b)

    def __mul__(self, b):
        return TermProduct(self, b)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        tau = torch.abs(as_tensor(tau))
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        k = torch.zeros_like(tau, dtype=torch.float64)
        tau = tau[..., None]

        if len(ar):
            k += torch.sum(ar * torch.exp(-cr * tau), axis=-1)

        if len(ac):
            arg = dc * tau
            k += torch.sum(
                torch.exp(-cc * tau)
                * (ac * torch.cos(arg) + bc * torch.sin(arg)),
                axis=-1,
            )

        return k

    def get_psd(self, omega):
        w2 = as_tensor(omega) ** 2
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        psd = torch.zeros_like(w2, dtype=torch.float64)
        w2 = w2[..., None]

        if len(ar):
            psd += torch.sum(ar * cr / (cr ** 2 + w2), axis=-1)

        if len(ac):
            w02 = cc ** 2 + dc ** 2
            psd += torch.sum(
                ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
                / (w2 ** 2 + 2.0 * (cc * cc - dc * dc) * w2 + w02 * w02),
                axis=-1,
            )

        return np.sqrt(2 / np.pi) * psd

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :]) + torch.diag(
            as_tensor(diag)
        )
        return K

    def get_celerite_matrices(self, x, diag):
        x = as_tensor(x)
        diag = as_tensor(diag)

        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        a = diag + torch.sum(ar) + torch.sum(ac)

        arg = dc[None, :] * x[:, None]
        cos = torch.cos(arg)
        sin = torch.sin(arg)
        z = torch.zeros_like(x, dtype=torch.float64)

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

    def get_conditional_mean_matrices(self, x, t):
        x = as_tensor(x)
        t = as_tensor(t)

        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        inds = ops.searchsorted(x, t)
        _, U_star, V_star, _ = self.get_celerite_matrices(t, t)

        c = np.concatenate([cr] + list(zip(cc, cc)))

        c = torch.cat((cr, cc, cc))
        mx = x.shape[0] - torch.ones_like(inds, dtype=torch.int64)
        dx = t - x[torch.min(inds, mx)]
        U_star *= torch.exp(-c[None, :] * dx[:, None])

        dx = (
            x[torch.max(inds - 1, torch.zeros_like(inds, dtype=torch.int64))]
            - t
        )
        V_star *= torch.exp(-c[None, :] * dx[:, None])

        return U_star, V_star, inds

    def dot(self, x, diag, y):
        a, U, V, P = self.get_celerite_matrices(x, diag)
        return ops.matmul(a, U, V, P, as_tensor(y))


class TermSum(Term):
    def __init__(self, *terms):
        super().__init__()
        if any(isinstance(term, IntegratedTerm) for term in terms):
            raise TypeError(
                "You cannot perform operations on an IntegratedTerm, it must "
                "be the outer term in the kernel"
            )
        self._terms = terms

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.get_coefficients() for t in self.terms)
        return tuple(torch.cat(a, axis=0) for a in zip(*coeffs))


class TermProduct(Term):
    def __init__(self, term1, term2):
        super().__init__()
        int1 = isinstance(term1, IntegratedTerm)
        int2 = isinstance(term2, IntegratedTerm)
        if int1 or int2:
            raise TypeError(
                "You cannot perform operations on an "
                "IntegratedTerm, it must be the outer term in "
                "the kernel"
            )
        self.term1 = term1
        self.term2 = term2

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        # First compute real terms
        ar = []
        cr = []
        ar.append(torch.flatten(c1[0][:, None] * c2[0][None, :]))
        cr.append(torch.flatten(c1[1][:, None] + c2[1][None, :]))

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        ac.append(torch.flatten(c1[0][:, None] * c2[2][None, :]))
        bc.append(torch.flatten(c1[0][:, None] * c2[3][None, :]))
        cc.append(torch.flatten(c1[1][:, None] + c2[4][None, :]))
        dc.append(
            torch.flatten(
                torch.zeros_like(c1[1], dtype=torch.float64)[:, None]
                + c2[5][None, :]
            )
        )

        ac.append(torch.flatten(c2[0][:, None] * c1[2][None, :]))
        bc.append(torch.flatten(c2[0][:, None] * c1[3][None, :]))
        cc.append(torch.flatten(c2[1][:, None] + c1[4][None, :]))
        dc.append(
            torch.flatten(
                torch.zeros_like(c2[1], dtype=torch.float64)[:, None]
                + c1[5][None, :]
            )
        )

        # complex * complex
        aj, bj, cj, dj = c1[2:]
        ak, bk, ck, dk = c2[2:]

        ac.append(
            torch.flatten(
                0.5 * (aj[:, None] * ak[None, :] + bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            torch.flatten(
                0.5 * (bj[:, None] * ak[None, :] - aj[:, None] * bk[None, :])
            )
        )
        cc.append(torch.flatten(cj[:, None] + ck[None, :]))
        dc.append(torch.flatten(dj[:, None] - dk[None, :]))

        ac.append(
            torch.flatten(
                0.5 * (aj[:, None] * ak[None, :] - bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            torch.flatten(
                0.5 * (bj[:, None] * ak[None, :] + aj[:, None] * bk[None, :])
            )
        )
        cc.append(torch.flatten(cj[:, None] + ck[None, :]))
        dc.append(torch.flatten(dj[:, None] + dk[None, :]))

        return [
            torch.cat(vals, axis=0)
            if len(vals)
            else torch.zeros(0, dtype=torch.float64)
            for vals in (ar, cr, ac, bc, cc, dc)
        ]


class TermDiff(Term):
    def __init__(self, term):
        super().__init__()
        if isinstance(term, IntegratedTerm):
            raise TypeError(
                "You cannot perform operations on an "
                "IntegratedTerm, it must be the outer term in "
                "the kernel"
            )
        self.term = term

    def get_coefficients(self):
        coeffs = self.term.get_coefficients()
        a, b, c, d = coeffs[2:]
        final_coeffs = [
            -coeffs[0] * coeffs[1] ** 2,
            coeffs[1],
            a * (d ** 2 - c ** 2) + 2 * b * c * d,
            b * (d ** 2 - c ** 2) - 2 * a * c * d,
            c,
            d,
        ]
        return final_coeffs


class IntegratedTerm(Term):
    def __init__(self, term, delta):
        super().__init__()
        self.term = term
        self.delta = as_tensor(delta)

    def get_celerite_matrices(self, x, diag):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real part
        cd = cr * dt
        delta_diag = 2 * torch.sum(ar * (cd - torch.sinh(cd)) / cd ** 2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = torch.sinh(cd)
        cosh = torch.cosh(cd)
        delta_diag += 2 * torch.sum(
            (
                C2 * cosh * torch.sin(dd)
                - C1 * sinh * torch.cos(dd)
                + (a * c + b * d) * dt * c2pd2
            )
            / norm
        )

        new_diag = as_tensor(diag) + delta_diag
        return super().get_celerite_matrices(x, new_diag)

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (torch.cosh(crd) - 1) / crd ** 2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c ** 2
        d2 = d ** 2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = torch.cosh(cd) * torch.cos(dd) - 1
        sin_term = torch.sinh(cd) * torch.sin(dd)

        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d

        coeffs += [
            factor * (C1 * cos_term - C2 * sin_term),
            factor * (C2 * cos_term + C1 * sin_term),
            c,
            d,
        ]

        return coeffs

    def get_psd(self, omega):
        psd0 = self.term.get_psd(omega)
        arg = 0.5 * self.delta * omega
        m = torch.abs(arg) > 1e-8
        sinc = torch.ones_like(arg)
        sinc[m] = torch.sin(arg) / arg
        return psd0 * sinc ** 2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Format the lags correctly
        tau0 = torch.abs(as_tensor(tau0))
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = torch.cosh(crd)
        norm = 2 * ar / crd ** 2
        K_large = torch.sum(norm * (cosh - 1) * torch.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + torch.sum(
            norm * (crdmt - torch.sinh(crdmt)), axis=-1
        )

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = torch.exp(-c * tau)
        cdt = torch.cos(d * tau)
        sdt = torch.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (torch.cosh(cd) * torch.cos(dd) - 1)
        sin_term = 2 * (torch.sinh(cd) * torch.sin(dd))
        factor = k0 * norm
        K_large += torch.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += torch.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = torch.exp(-c * dmt)
        edpt = torch.exp(-c * dpt)
        cos_term = (
            edmt * torch.cos(d * dmt)
            + edpt * torch.cos(d * dpt)
            - 2 * k0 * cdt
        )
        sin_term = (
            edmt * torch.sin(d * dmt)
            + edpt * torch.sin(d * dpt)
            - 2 * k0 * sdt
        )
        K_small += torch.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += torch.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        mask = tau0 >= dt
        return K_large * mask + K_small * (~mask)


class RealTerm(Term):
    def __init__(self, *, a, c):
        super().__init__()
        self.a = as_tensor(a)
        self.c = as_tensor(c)

    def get_coefficients(self):
        e = as_tensor([])
        return (
            self.a[None],
            self.c[None],
            e,
            e,
            e,
            e,
        )


class ComplexTerm(Term):
    def __init__(self, *, a, b, c, d):
        super().__init__()
        self.a = as_tensor(a)
        self.b = as_tensor(b)
        self.c = as_tensor(c)
        self.d = as_tensor(d)

    def get_coefficients(self):
        e = as_tensor([])
        return (
            e,
            e,
            self.a[None],
            self.b[None],
            self.c[None],
            self.d[None],
        )


class SHOTerm(Term):
    def __init__(self, *, w0, Q, S0=None, Sw4=None, S_tot=None, eps=1e-5):
        super().__init__()
        self.eps = as_tensor(eps)
        self.w0 = as_tensor(w0)
        self.Q = as_tensor(Q)

        self.S0 = None if S0 is None else as_tensor(S0)
        self.Sw4 = None if Sw4 is None else as_tensor(Sw4)
        self.S_tot = None if S_tot is None else as_tensor(S_tot)

    def _get_S0(self):
        if self.S0 is not None:
            if self.Sw4 is not None or self.S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            return self.S0
        elif self.Sw4 is not None:
            if self.S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            return self.Sw4 / self.w0 ** 4
        elif self.S_tot is not None:
            return self.S_tot / (self.w0 * self.Q)
        raise ValueError("one of S0, Sw4, and S_tot must be given")

    def overdamped(self):
        Q = self.Q
        f = torch.sqrt(torch.max(1.0 - 4.0 * Q ** 2, self.eps))
        e = as_tensor([])
        return (
            0.5
            * self._get_S0()
            * self.w0
            * Q
            * torch.stack([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * torch.stack([1.0 - f, 1.0 + f]),
            e,
            e,
            e,
            e,
        )

    def underdamped(self):
        Q = self.Q
        f = torch.sqrt(torch.max(4.0 * Q ** 2 - 1.0, self.eps))
        a = self._get_S0() * self.w0 * Q
        c = 0.5 * self.w0 / Q
        e = as_tensor([])
        return (
            e,
            e,
            a[None],
            a[None] / f,
            c[None],
            c[None] * f,
        )

    def get_coefficients(self):
        return self.overdamped() if self.Q < 0.5 else self.underdamped()


class Matern32Term(Term):
    def __init__(self, *, sigma, rho, eps=0.01):
        super().__init__()
        self.sigma = as_tensor(sigma)
        self.rho = as_tensor(rho)
        self.eps = as_tensor(eps)

    def get_coefficients(self):
        w0 = np.sqrt(3.0) / self.rho
        S0 = self.sigma ** 2 / w0
        empty = as_tensor([])
        return (
            empty,
            empty,
            w0[None] * S0,
            w0[None] * (w0 * S0 / self.eps),
            w0[None],
            self.eps[None],
        )


class RotationTerm(TermSum):
    def __init__(self, *, amp, Q0, deltaQ, period, mix, **kwargs):
        amp = as_tensor(amp)
        Q0 = as_tensor(Q0)
        deltaQ = as_tensor(deltaQ)
        period = as_tensor(period)
        mix = as_tensor(mix)

        # One term with a period of period
        Q1 = 0.5 + Q0 + deltaQ
        w1 = 4 * np.pi * Q1 / (period * torch.sqrt(4 * Q1 ** 2 - 1))
        S1 = amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + Q0
        w2 = 8 * np.pi * Q2 / (period * torch.sqrt(4 * Q2 ** 2 - 1))
        S2 = mix * amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2)
        )

        self.amp = amp
        self.Q0 = Q0
        self.deltaQ = deltaQ
        self.period = period
        self.mix = mix
