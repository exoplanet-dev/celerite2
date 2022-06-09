# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
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

from jax import numpy as np

import celerite2.terms as base_terms
from celerite2.jax import ops


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
        coeff = self.get_coefficients()
        r = self._get_value_real(coeff[:2], tau)
        c = self._get_value_complex(coeff[2:], tau)
        return r + c

    def _get_value_real(self, coefficients, tau):
        ar, cr = coefficients
        tau = np.abs(np.atleast_1d(tau))
        tau = tau[..., None]
        return np.sum(ar * np.exp(-cr * tau), axis=-1)

    def _get_value_complex(self, coefficients, tau):
        ac, bc, cc, dc = coefficients
        tau = np.abs(np.atleast_1d(tau))
        tau = tau[..., None]
        arg = dc * tau
        return np.sum(
            np.exp(-cc * tau) * (ac * np.cos(arg) + bc * np.sin(arg)),
            axis=-1,
        )

    def get_psd(self, omega):
        coeff = self.get_coefficients()
        r = self._get_psd_real(coeff[:2], omega)
        c = self._get_psd_complex(coeff[2:], omega)
        return r + c

    def _get_psd_real(self, coefficients, omega):
        ar, cr = coefficients
        w2 = np.atleast_1d(omega) ** 2
        w2 = w2[..., None]
        psd = np.sum(ar * cr / (cr**2 + w2), axis=-1)
        return np.sqrt(2 / np.pi) * psd

    def _get_psd_complex(self, coefficients, omega):
        ac, bc, cc, dc = coefficients
        w2 = np.atleast_1d(omega) ** 2
        w2 = w2[..., None]
        w02 = cc**2 + dc**2
        psd = np.sum(
            ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
            / (w2**2 + 2.0 * (cc * cc - dc * dc) * w2 + w02 * w02),
            axis=-1,
        )
        return np.sqrt(2 / np.pi) * psd

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        return K + np.diag(diag)

    def get_celerite_matrices(self, x, diag, **kwargs):
        x = np.atleast_1d(x)
        diag = np.atleast_1d(diag)
        if x.shape != diag.shape:
            raise ValueError("dimension mismatch")
        coeff = self.get_coefficients()
        cr, ar, Ur, Vr = self._get_celerite_matrices_real(
            coeff[:2], x, **kwargs
        )
        cc, ac, Uc, Vc = self._get_celerite_matrices_complex(
            coeff[2:], x, **kwargs
        )
        c = np.concatenate((cr, cc))
        a = diag + ar + ac
        U = np.concatenate((Ur, Uc), axis=1)
        V = np.concatenate((Vr, Vc), axis=1)
        return c, a, U, V

    def _get_celerite_matrices_real(self, coefficients, x, **kwargs):
        if len(x.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
        ar, cr = coefficients
        z = np.zeros_like(x)
        return (
            cr,
            np.sum(ar),
            ar[None, :] + z[:, None],
            np.ones_like(ar)[None, :] + z[:, None],
        )

    def _get_celerite_matrices_complex(self, coefficients, x, **kwargs):
        if len(x.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
        ac, bc, cc, dc = coefficients
        arg = dc[None, :] * x[:, None]
        cos = np.cos(arg)
        sin = np.sin(arg)
        U = np.concatenate(
            (
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )
        V = np.concatenate((cos, sin), axis=1)
        c = np.concatenate((cc, cc))
        return c, np.sum(ac), U, V

    def dot(self, x, diag, y):
        y = np.atleast_1d(y)

        is_vector = False
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("'y' can only be a vector or matrix")

        c, a, U, V = self.get_celerite_matrices(x, diag)
        z = y * a[:, None]
        z += ops.matmul_lower(x, c, U, V, y)
        z += ops.matmul_upper(x, c, U, V, y)

        if is_vector:
            return z[:, 0]
        return z


class TermSum(Term):
    def __init__(self, *terms):
        # if any(isinstance(term, TermConvolution) for term in terms):
        #     raise TypeError(
        #         "You cannot perform operations on an TermConvolution, it must "
        #         "be the outer term in the kernel"
        #     )
        self._terms = terms

    @property
    def terms(self):
        return self._terms

    # def get_coefficients(self):
    #     coeffs = (t.get_coefficients() for t in self.terms)
    #     return tuple(np.concatenate(c) for c in zip(*coeffs))

    def get_value(self, tau):
        tau = np.atleast_1d(tau)
        return sum(term.get_value(tau) for term in self._terms)

    def get_psd(self, omega):
        omega = np.atleast_1d(omega)
        return sum(term.get_psd(omega) for term in self._terms)

    def get_celerite_matrices(self, x, diag, **kwargs):
        diag = np.atleast_1d(diag)
        matrices = (
            term.get_celerite_matrices(x, np.zeros_like(diag), **kwargs)
            for term in self._terms
        )
        c, a, U, V = zip(*matrices)
        return (
            np.concatenate(c, axis=-1),
            sum(a) + diag,
            np.concatenate(U, axis=-1),
            np.concatenate(V, axis=-1),
        )


class TermProduct(Term):
    def __init__(self, term1, term2):
        # int1 = isinstance(term1, TermConvolution)
        # int2 = isinstance(term2, TermConvolution)
        # if int1 or int2:
        #     raise TypeError(
        #         "You cannot perform operations on an TermConvolution, it must "
        #         "be the outer term in the kernel"
        #     )
        self.term1 = term1
        self.term2 = term2

    def get_value(self, tau):
        tau = np.atleast_1d(tau)
        return self.term1.get_value(tau) * self.term2.get_value(tau)

    def get_psd(self, omega):
        raise NotImplementedError(
            "The PSD function is not implemented for general Term products"
        )

    def get_celerite_matrices(self, x, diag, **kwargs):
        diag = np.atleast_1d(diag)
        z = np.zeros_like(diag)
        c1, a1, U1, V1 = self.term1.get_celerite_matrices(x, z, **kwargs)
        c2, a2, U2, V2 = self.term2.get_celerite_matrices(x, z, **kwargs)

        mg = np.meshgrid(np.arange(c1.shape[0]), np.arange(c2.shape[0]))
        i = mg[0].flatten()
        j = mg[1].flatten()

        c = c1[i] + c2[j]
        a = a1 * a2 + diag
        U = U1[:, i] * U2[:, j]
        V = V1[:, i] * V2[:, j]
        return c, a, U, V

    # def get_coefficients(self):
    #     c1 = self.term1.get_coefficients()
    #     c2 = self.term2.get_coefficients()

    #     # First compute real terms
    #     ar = []
    #     cr = []
    #     gen = product(zip(c1[0], c1[1]), zip(c2[0], c2[1]))
    #     for i, ((aj, cj), (ak, ck)) in enumerate(gen):
    #         ar.append(aj * ak)
    #         cr.append(cj + ck)

    #     # Then the complex terms
    #     ac = []
    #     bc = []
    #     cc = []
    #     dc = []

    #     # real * complex
    #     gen = product(zip(c1[0], c1[1]), zip(*(c2[2:])))
    #     gen = chain(gen, product(zip(c2[0], c2[1]), zip(*(c1[2:]))))
    #     for i, ((aj, cj), (ak, bk, ck, dk)) in enumerate(gen):
    #         ac.append(aj * ak)
    #         bc.append(aj * bk)
    #         cc.append(cj + ck)
    #         dc.append(dk)

    #     # complex * complex
    #     gen = product(zip(*(c1[2:])), zip(*(c2[2:])))
    #     for i, ((aj, bj, cj, dj), (ak, bk, ck, dk)) in enumerate(gen):
    #         ac.append(0.5 * (aj * ak + bj * bk))
    #         bc.append(0.5 * (bj * ak - aj * bk))
    #         cc.append(cj + ck)
    #         dc.append(dj - dk)

    #         ac.append(0.5 * (aj * ak - bj * bk))
    #         bc.append(0.5 * (bj * ak + aj * bk))
    #         cc.append(cj + ck)
    #         dc.append(dj + dk)

    #     return list(map(np.array, (ar, cr, ac, bc, cc, dc)))


class TermDiff(Term):
    def __init__(self, term):
        try:
            self.coefficients = term.get_coefficients()
        except NotImplementedError:
            raise TypeError(
                "Term operations can only be performed on terms that provide "
                "coefficients"
            )

    def get_coefficients(self):
        coeffs = self.coefficients
        a, b, c, d = coeffs[2:]
        final_coeffs = [
            -coeffs[0] * coeffs[1] ** 2,
            coeffs[1],
            a * (d**2 - c**2) + 2 * b * c * d,
            b * (d**2 - c**2) - 2 * a * c * d,
            c,
            d,
        ]
        return final_coeffs


class TermConvolution(Term):
    def __init__(self, term, delta):
        self.delta = np.float64(delta)
        try:
            self.coefficients = term.get_coefficients()
        except NotImplementedError:
            raise TypeError(
                "Term operations can only be performed on terms that provide "
                "coefficients"
            )

    def get_celerite_matrices(self, x, diag, **kwargs):
        dt = self.delta
        ar, cr, a, b, c, d = self.coefficients

        # Real part
        cd = cr * dt
        delta_diag = 2 * np.sum(ar * (cd - np.sinh(cd)) / cd**2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c**2
        d2 = d**2
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
        ar, cr, a, b, c, d = self.coefficients

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (np.cosh(crd) - 1) / crd**2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c**2
        d2 = d**2
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
        psd0 = super()._get_psd_real(self.coefficients[:2], omega)
        psd0 += super()._get_psd_complex(self.coefficients[2:], omega)
        arg = 0.5 * self.delta * omega
        arg += 1e-8 * (np.abs(arg) < 1e-8) * np.sign(arg)
        sinc = np.sin(arg) / arg
        return psd0 * sinc**2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.coefficients

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
        norm = 2 * ar / crd**2
        K_large = np.sum(norm * (cosh - 1) * np.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + np.sum(norm * (crdmt - np.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c**2
        d2 = d**2
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


def SHOTerm(*args, **kwargs):
    over = OverdampedSHOTerm(*args, **kwargs)
    under = UnderdampedSHOTerm(*args, **kwargs)
    if over.Q < 0.5:
        return over
    return under


class SHOTerm(Term):
    __parameter_spec__ = base_terms.SHOTerm.__parameter_spec__

    @base_terms.handle_parameter_spec(np.float64)
    def __init__(self, *, eps=1e-5, **kwargs):
        self.eps = np.float64(eps)

    def get_overdamped_coefficients(self):
        Q = self.Q
        f = np.sqrt(np.maximum(1.0 - 4.0 * Q**2, self.eps))
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * np.array([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * np.array([1.0 - f, 1.0 + f]),
        )

    def get_underdamped_coefficients(self):
        Q = self.Q
        f = np.sqrt(np.maximum(4.0 * Q**2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        return (
            np.array([a]),
            np.array([a / f]),
            np.array([c]),
            np.array([c * f]),
        )

    def get_value(self, tau):
        return np.where(
            np.less(self.Q, 0.5),
            super()._get_value_real(self.get_overdamped_coefficients(), tau),
            super()._get_value_complex(
                self.get_underdamped_coefficients(), tau
            ),
        )

    def get_psd(self, omega):
        return np.where(
            np.less(self.Q, 0.5),
            super()._get_psd_real(self.get_overdamped_coefficients(), omega),
            super()._get_psd_complex(
                self.get_underdamped_coefficients(), omega
            ),
        )

    def get_celerite_matrices(self, x, diag, **kwargs):
        x = np.atleast_1d(x)
        diag = np.atleast_1d(diag)
        if x.shape != diag.shape:
            raise ValueError("dimension mismatch")
        cr, ar, Ur, Vr = super()._get_celerite_matrices_real(
            self.get_overdamped_coefficients(), x, **kwargs
        )
        cc, ac, Uc, Vc = super()._get_celerite_matrices_complex(
            self.get_underdamped_coefficients(), x, **kwargs
        )
        ar = ar + diag
        ac = ac + diag
        cond = np.less(self.Q, 0.5)
        return [
            np.where(cond, a, b)
            for a, b in zip((cr, ar, Ur, Vr), (cc, ac, Uc, Vc))
        ]


OverdampedSHOTerm = SHOTerm
UnderdampedSHOTerm = SHOTerm


class Matern32Term(Term):
    def __init__(self, *, sigma, rho, eps=0.01):
        self.sigma = np.float64(sigma)
        self.rho = np.float64(rho)
        self.eps = np.float64(eps)

    def get_coefficients(self):
        w0 = np.sqrt(3) / self.rho
        S0 = self.sigma**2 / w0
        e = np.empty(0)
        return (
            e,
            e,
            np.array([w0 * S0]),
            np.array([w0**2 * S0 / self.eps]),
            np.array([w0]),
            np.array([self.eps]),
        )


class RotationTerm(TermSum):
    def __init__(self, *, sigma, period, Q0, dQ, f):
        self.sigma = np.float64(sigma)
        self.period = np.float64(period)
        self.Q0 = np.float64(Q0)
        self.dQ = np.float64(dQ)
        self.f = np.float64(f)

        self.amp = self.sigma**2 / (1 + self.f)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1**2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2**2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            UnderdampedSHOTerm(S0=S1, w0=w1, Q=Q1),
            UnderdampedSHOTerm(S0=S2, w0=w2, Q=Q2),
        )
