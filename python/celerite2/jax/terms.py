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
    "OriginalCeleriteTerm",
]
from itertools import chain, product

from jax import numpy as np

from . import ops


class Term:
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
        tau = np.abs(np.atleast_1d(tau))
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        k = np.zeros_like(tau)
        tau = tau[..., None]

        if len(ar):
            k += np.sum(ar * np.exp(-cr * tau), axis=-1)

        if len(ac):
            arg = dc * tau
            k += np.sum(
                np.exp(-cc * tau) * (ac * np.cos(arg) + bc * np.sin(arg)),
                axis=-1,
            )

        return k

    def get_psd(self, omega):
        w2 = np.atleast_1d(omega) ** 2
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        psd = np.zeros_like(w2)
        w2 = w2[..., None]

        if len(ar):
            psd += np.sum(ar * cr / (cr ** 2 + w2), axis=-1)

        if len(ac):
            w02 = cc ** 2 + dc ** 2
            psd += np.sum(
                ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
                / (w2 ** 2 + 2.0 * (cc * cc - dc * dc) * w2 + w02 * w02),
                axis=-1,
            )

        return np.sqrt(2 / np.pi) * psd

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        return K + np.diag(diag)

    def get_celerite_matrices(
        self, x, diag, *, a=None, U=None, V=None, P=None
    ):
        x = np.atleast_1d(x)
        diag = np.atleast_1d(diag)
        if len(x.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
        if x.shape != diag.shape:
            raise ValueError("dimension mismatch")

        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        a = diag + np.sum(ar) + np.sum(ac)

        arg = dc[None, :] * x[:, None]
        cos = np.cos(arg)
        sin = np.sin(arg)
        z = np.zeros_like(x)

        U = np.concatenate(
            (
                ar[None, :] + z[:, None],
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )

        V = np.concatenate(
            (np.ones_like(ar)[None, :] + z[:, None], cos, sin), axis=1,
        )

        dx = x[1:] - x[:-1]
        c = np.concatenate((cr, cc, cc))
        P = np.exp(-c[None, :] * dx[:, None])

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        inds = np.searchsorted(x, t)
        _, U_star, V_star, _ = self.get_celerite_matrices(t, t)

        c = np.concatenate((cr, cc, cc))

        dx = t - x[np.minimum(inds, x.size - 1)]
        U_star *= np.exp(-c[None, :] * dx[:, None])

        dx = x[np.maximum(inds - 1, 0)] - t
        V_star *= np.exp(-c[None, :] * dx[:, None])

        return U_star, V_star, inds

    def dot(self, x, diag, y):
        a, U, V, P = self.get_celerite_matrices(x, diag)
        y = np.atleast_1d(y)
        if y.shape[0] != len(a):
            raise ValueError("dimension mismatch")
        return ops.matmul(a, U, V, P, y)


class TermSum(Term):
    def __init__(self, *terms):
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
        return tuple(np.concatenate(c) for c in zip(*coeffs))


class TermProduct(Term):
    def __init__(self, term1, term2):
        int1 = isinstance(term1, IntegratedTerm)
        int2 = isinstance(term2, IntegratedTerm)
        if int1 or int2:
            raise TypeError(
                "You cannot perform operations on an IntegratedTerm, it must "
                "be the outer term in the kernel"
            )
        self.term1 = term1
        self.term2 = term2

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        # First compute real terms
        ar = []
        cr = []
        gen = product(zip(c1[0], c1[1]), zip(c2[0], c2[1]))
        for i, ((aj, cj), (ak, ck)) in enumerate(gen):
            ar.append(aj * ak)
            cr.append(cj + ck)

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        gen = product(zip(c1[0], c1[1]), zip(*(c2[2:])))
        gen = chain(gen, product(zip(c2[0], c2[1]), zip(*(c1[2:]))))
        for i, ((aj, cj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(aj * ak)
            bc.append(aj * bk)
            cc.append(cj + ck)
            dc.append(dk)

        # complex * complex
        gen = product(zip(*(c1[2:])), zip(*(c2[2:])))
        for i, ((aj, bj, cj, dj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(0.5 * (aj * ak + bj * bk))
            bc.append(0.5 * (bj * ak - aj * bk))
            cc.append(cj + ck)
            dc.append(dj - dk)

            ac.append(0.5 * (aj * ak - bj * bk))
            bc.append(0.5 * (bj * ak + aj * bk))
            cc.append(cj + ck)
            dc.append(dj + dk)

        return list(map(np.array, (ar, cr, ac, bc, cc, dc)))


class TermDiff(Term):
    def __init__(self, term):
        if isinstance(term, IntegratedTerm):
            raise TypeError(
                "You cannot perform operations on an IntegratedTerm, it must "
                "be the outer term in the kernel"
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
        self.term = term
        self.delta = np.float64(delta)

    def get_celerite_matrices(
        self, x, diag, *, a=None, U=None, V=None, P=None
    ):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real part
        cd = cr * dt
        delta_diag = 2 * np.sum(ar * (cd - np.sinh(cd)) / cd ** 2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = np.sinh(cd)
        cosh = np.cosh(cd)
        delta_diag += 2 * np.sum(
            (
                C2 * cosh * np.sin(dd)
                - C1 * sinh * np.cos(dd)
                + (a * c + b * d) * dt * c2pd2
            )
            / norm
        )

        new_diag = diag + delta_diag

        return super().get_celerite_matrices(x, new_diag, a=a, U=U, V=V, P=P)

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (np.cosh(crd) - 1) / crd ** 2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c ** 2
        d2 = d ** 2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = np.cosh(cd) * np.cos(dd) - 1
        sin_term = np.sinh(cd) * np.sin(dd)

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
        omega = np.atleast_1d(omega)
        psd0 = self.term.get_psd(omega)
        arg = 0.5 * self.delta * omega
        arg += 1e-8 * (np.abs(arg) < 1e-8) * np.sign(arg)
        sinc = np.sin(arg) / arg
        return psd0 * sinc ** 2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Format the lags correctly
        tau0 = np.abs(np.atleast_1d(tau0))
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = np.cosh(crd)
        norm = 2 * ar / crd ** 2
        K_large = np.sum(norm * (cosh - 1) * np.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + np.sum(norm * (crdmt - np.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = np.exp(-c * tau)
        cdt = np.cos(d * tau)
        sdt = np.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (np.cosh(cd) * np.cos(dd) - 1)
        sin_term = 2 * (np.sinh(cd) * np.sin(dd))
        factor = k0 * norm
        K_large += np.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += np.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = np.exp(-c * dmt)
        edpt = np.exp(-c * dpt)
        cos_term = (
            edmt * np.cos(d * dmt) + edpt * np.cos(d * dpt) - 2 * k0 * cdt
        )
        sin_term = (
            edmt * np.sin(d * dmt) + edpt * np.sin(d * dpt) - 2 * k0 * sdt
        )
        K_small += np.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += np.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        mask = tau0 >= dt
        return K_large * mask + K_small * (~mask)


class RealTerm(Term):
    def __init__(self, *, a, c):
        self.a = np.float64(a)
        self.c = np.float64(c)

    def get_coefficients(self):
        e = np.empty(0)
        return np.array([self.a]), np.array([self.c]), e, e, e, e


class ComplexTerm(Term):
    def __init__(self, *, a, b, c, d):
        self.a = np.float64(a)
        self.b = np.float64(b)
        self.c = np.float64(c)
        self.d = np.float64(d)

    def get_coefficients(self):
        e = np.empty(0)
        return (
            e,
            e,
            np.array([self.a]),
            np.array([self.b]),
            np.array([self.c]),
            np.array([self.d]),
        )


class SHOTerm(Term):
    def __init__(self, *, w0, Q, S0=None, Sw4=None, S_tot=None, eps=1e-5):
        self.eps = np.float64(eps)
        self.w0 = np.float64(w0)
        self.Q = np.float64(Q)

        if S0 is not None:
            if Sw4 is not None or S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            self.S0 = np.float64(S0)
        elif Sw4 is not None:
            if S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            self.S0 = np.float64(Sw4) / self.w0 ** 4
        elif S_tot is not None:
            self.S0 = np.float64(S_tot) / (self.w0 * self.Q)
        else:
            raise ValueError("one of S0, Sw4, and S_tot must be given")

    def overdamped(self):
        Q = self.Q
        f = np.sqrt(np.maximum(1.0 - 4.0 * Q ** 2, self.eps))
        e = np.empty(0)
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * np.array([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * np.array([1.0 - f, 1.0 + f]),
            e,
            e,
            e,
            e,
        )

    def underdamped(self):
        Q = self.Q
        f = np.sqrt(np.maximum(4.0 * Q ** 2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        e = np.empty(0)
        return (
            e,
            e,
            np.array([a]),
            np.array([a / f]),
            np.array([c]),
            np.array([c * f]),
        )

    def get_coefficients(self):
        return self.overdamped() if self.Q < 0.5 else self.underdamped()


class Matern32Term(Term):
    def __init__(self, *, sigma, rho, eps=0.01):
        self.sigma = np.float64(sigma)
        self.rho = np.float64(rho)
        self.eps = np.float64(eps)

    def get_coefficients(self):
        w0 = np.sqrt(3) / self.rho
        S0 = self.sigma ** 2 / w0
        e = np.empty(0)
        return (
            e,
            e,
            np.array([w0 * S0]),
            np.array([w0 ** 2 * S0 / self.eps]),
            np.array([w0]),
            np.array([self.eps]),
        )


class RotationTerm(TermSum):
    def __init__(self, *, amp, Q0, deltaQ, period, mix):
        self.amp = np.float64(amp)
        self.Q0 = np.float64(Q0)
        self.deltaQ = np.float64(deltaQ)
        self.period = np.float64(period)
        self.mix = np.float64(mix)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.deltaQ
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1 ** 2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2 ** 2 - 1))
        S2 = self.mix * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2)
        )


class OriginalCeleriteTerm(Term):
    def __init__(self, term):
        self.term = term

    def get_coefficients(self):
        return self.term.get_all_coefficients()
