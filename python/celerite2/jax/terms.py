# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
    "TermSumGeneral",
    "TermProduct",
    "TermDiff",
    "TermConvolution",
    "RealTerm",
    "ComplexTerm",
    "SHOTerm",
    "OverdampedSHOTerm",
    "UnderdampedSHOTerm",
    "Matern32Term",
    "RotationTerm",
]
from itertools import chain, product

from jax import numpy as np

from .. import terms as base_terms
from . import ops


class Term:
    __doc__ = base_terms.Term.__doc__
    __requires_general_addition__ = False

    def __add__(self, b):
        terms = tuple(self.terms) + (b,)
        if (
            self.__requires_general_addition__
            or b.__requires_general_addition__
        ):
            return TermSumGeneral(*terms)
        return TermSum(*terms)

    def __mul__(self, b):
        return TermProduct(self, b)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, t, t2=None, *, diag=None, X=None, X2=None):
        t = np.atleast_1d(t)
        if diag is None:
            diag = np.zeros_like(t)
        c1, a1, U1, V1 = self.get_celerite_matrices(t, diag, X=X)
        if t2 is None:
            Y = np.eye(t.shape[0])
            K = np.diag(a1)
            K += ops.matmul_lower(t, c1, U1, V1, Y)
            K += ops.matmul_upper(t, c1, U1, V1, Y)
            return K

        t2 = np.atleast_1d(t2)
        c2, a2, U2, V2 = self.get_celerite_matrices(
            t2, np.zeros_like(t2), X=X2
        )
        Y = np.eye(t2.shape[0])
        K = np.zeros((t.shape[0], t2.shape[0]))
        K += ops.general_matmul_lower(t, t2, c1, U1, V2, Y)
        K += ops.general_matmul_upper(t, t2, c1, V1, U2, Y)
        return K

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

    def get_celerite_matrices(self, t, diag, **kwargs):
        t = np.atleast_1d(t)
        diag = np.atleast_1d(diag)
        if t.ndim != 1:
            raise ValueError("'t' must be one-dimensional")
        if t.shape != diag.shape:
            raise ValueError("dimension mismatch")

        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        a = diag + np.sum(ar) + np.sum(ac)

        arg = dc[None, :] * t[:, None]
        cos = np.cos(arg)
        sin = np.sin(arg)
        z = np.zeros_like(t)

        U = np.concatenate(
            (
                ar[None, :] + z[:, None],
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )

        V = np.concatenate(
            (np.ones_like(ar)[None, :] + z[:, None], cos, sin),
            axis=1,
        )

        c = np.concatenate((cr, cc, cc))

        return c, a, U, V

    def dot(self, t, diag, y, *, X=None):
        y = np.atleast_1d(y)

        is_vector = False
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("'y' can only be a vector or matrix")

        c, a, U, V = self.get_celerite_matrices(t, diag, X=X)
        z = y * a[:, None]
        z += ops.matmul_lower(t, c, U, V, y)
        z += ops.matmul_upper(t, c, U, V, y)

        if is_vector:
            return z[:, 0]
        return z


class TermSum(Term):
    __doc__ = base_terms.TermSum.__doc__

    def __init__(self, *terms):
        if any(term.__requires_general_addition__ for term in terms):
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
            )
        self._terms = terms

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.get_coefficients() for t in self.terms)
        return tuple(np.concatenate(c) for c in zip(*coeffs))


class TermSumGeneral(Term):
    __doc__ = base_terms.TermSumGeneral.__doc__
    __requires_general_addition__ = True

    def __init__(self, *terms):
        basic = [
            term for term in terms if not term.__requires_general_addition__
        ]
        self._terms = [
            term for term in terms if term.__requires_general_addition__
        ]
        if len(basic):
            self._terms.insert(0, TermSum(*basic))
        if not len(self._terms):
            raise ValueError(
                "A general term sum cannot be instantiated without any terms"
            )

    @property
    def terms(self):
        return self._terms

    def get_value(self, *args, **kwargs):
        K = self.terms[0].get_value(*args, **kwargs)
        for term in self.terms[1:]:
            K += term.get_value(*args, **kwargs)
        return K

    def get_psd(self, omega):
        p = self.terms[0].get_psd(omega)
        for term in self.terms[1:]:
            p += term.get_psd(omega)
        return p

    def get_celerite_matrices(
        self, t, diag, *, c=None, a=None, U=None, V=None, X=None
    ):
        diag = np.atleast_1d(diag)
        matrices = [
            term.get_celerite_matrices(t, np.zeros_like(diag), X=X)
            for term in self.terms
        ]
        c = np.concatenate([m[0] for m in matrices])
        a = diag + np.sum([m[1] for m in matrices], axis=0)
        U = np.concatenate([m[2] for m in matrices], axis=1)
        V = np.concatenate([m[3] for m in matrices], axis=1)
        return c, a, U, V


class TermProduct(Term):
    __doc__ = base_terms.TermProduct.__doc__

    def __init__(self, term1, term2):
        if (
            term1.__requires_general_addition__
            or term2.__requires_general_addition__
        ):
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
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
    __doc__ = base_terms.TermDiff.__doc__

    def __init__(self, term):
        if term.__requires_general_addition__:
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
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


class TermConvolution(Term):
    __doc__ = base_terms.TermConvolution.__doc__
    __requires_general_addition__ = True

    def __init__(self, term, delta):
        if term.__requires_general_addition__:
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
            )
        self.term = term
        self.delta = np.float64(delta)

    def get_celerite_matrices(self, x, diag, **kwargs):
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

        return super().get_celerite_matrices(x, new_diag)

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


class RealTerm(Term):
    __doc__ = base_terms.RealTerm.__doc__

    def __init__(self, *, a, c):
        self.a = np.float64(a)
        self.c = np.float64(c)

    def get_coefficients(self):
        e = np.empty(0)
        return np.array([self.a]), np.array([self.c]), e, e, e, e


class ComplexTerm(Term):
    __doc__ = base_terms.ComplexTerm.__doc__

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


def SHOTerm(*args, **kwargs):
    over = OverdampedSHOTerm(*args, **kwargs)
    under = UnderdampedSHOTerm(*args, **kwargs)
    if over.Q < 0.5:
        return over
    return under


SHOTerm.__doc__ = base_terms.SHOTerm.__doc__


class OverdampedSHOTerm(Term):
    __parameter_spec__ = base_terms.SHOTerm.__parameter_spec__

    @base_terms.handle_parameter_spec(np.float64)
    def __init__(self, *, eps=1e-5, **kwargs):
        self.eps = np.float64(eps)

    def get_coefficients(self):
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


class UnderdampedSHOTerm(Term):
    __parameter_spec__ = base_terms.SHOTerm.__parameter_spec__

    @base_terms.handle_parameter_spec(np.float64)
    def __init__(self, *, eps=1e-5, **kwargs):
        self.eps = np.float64(eps)

    def get_coefficients(self):
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


class Matern32Term(Term):
    __doc__ = base_terms.Matern32Term.__doc__

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
    __doc__ = base_terms.RotationTerm.__doc__

    def __init__(self, *, sigma, period, Q0, dQ, f):
        self.sigma = np.float64(sigma)
        self.period = np.float64(period)
        self.Q0 = np.float64(Q0)
        self.dQ = np.float64(dQ)
        self.f = np.float64(f)

        self.amp = self.sigma ** 2 / (1 + self.f)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1 ** 2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2 ** 2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            UnderdampedSHOTerm(S0=S1, w0=w1, Q=Q1),
            UnderdampedSHOTerm(S0=S2, w0=w2, Q=Q2),
        )
