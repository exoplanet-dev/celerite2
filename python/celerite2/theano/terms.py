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
    "Matern32Term",
    "RotationTerm",
]
import numpy as np
import theano
from theano import tensor as tt
from theano.ifelse import ifelse

from .. import terms as base_terms
from . import ops


class Term(base_terms.Term):
    __doc__ = base_terms.Term.__doc__

    def __init__(self, *, dtype="float64"):
        self.dtype = dtype
        self.coefficients = self.get_coefficients()

    def __add__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermSum(self, b, dtype=dtype)

    def __mul__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(self, b, dtype=dtype)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        ar, cr, ac, bc, cc, dc = self.coefficients
        tau = tt.abs_(tau)
        tau = tt.shape_padright(tau)
        K = tt.sum(ar * tt.exp(-cr * tau), axis=-1)
        factor = tt.exp(-cc * tau)
        K += tt.sum(ac * factor * tt.cos(dc * tau), axis=-1)
        K += tt.sum(bc * factor * tt.sin(dc * tau), axis=-1)
        return K

    def get_psd(self, omega):
        ar, cr, ac, bc, cc, dc = self.coefficients
        omega = tt.shape_padright(omega)
        w2 = omega ** 2
        w02 = cc ** 2 + dc ** 2
        power = tt.sum(ar * cr / (cr ** 2 + w2), axis=-1)
        power += tt.sum(
            ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
            / (w2 * w2 + 2.0 * (cc ** 2 - dc ** 2) * w2 + w02 * w02),
            axis=-1,
        )
        return np.sqrt(2.0 / np.pi) * power

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        K += tt.diag(diag)
        return K

    def get_celerite_matrices(self, x, diag):
        x = tt.as_tensor_variable(x)
        diag = tt.as_tensor_variable(diag)
        ar, cr, ac, bc, cc, dc = self.coefficients
        a = diag + tt.sum(ar) + tt.sum(ac)

        arg = dc[None, :] * x[:, None]
        cos = tt.cos(arg)
        sin = tt.sin(arg)
        z = tt.zeros_like(x)

        U = tt.concatenate(
            (
                ar[None, :] + z[:, None],
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )

        V = tt.concatenate(
            (tt.ones_like(ar)[None, :] + z[:, None], cos, sin),
            axis=1,
        )

        dx = x[1:] - x[:-1]
        c = tt.concatenate((cr, cc, cc))
        P = tt.exp(-c[None, :] * dx[:, None])

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        ar, cr, ac, bc, cc, dc = self.coefficients
        x = tt.as_tensor_variable(x)

        inds = tt.extra_ops.searchsorted(x, t)
        _, U_star, V_star, _ = self.get_celerite_matrices(t, t)

        c = tt.concatenate((cr, cc, cc))
        dx = t - x[tt.minimum(inds, x.size - 1)]
        U_star *= tt.exp(-c[None, :] * dx[:, None])

        dx = x[tt.maximum(inds - 1, 0)] - t
        V_star *= tt.exp(-c[None, :] * dx[:, None])

        return U_star, V_star, inds

    def dot(self, x, diag, y):
        a, U, V, P = self.get_celerite_matrices(x, diag)
        return ops.matmul(a, U, V, P, tt.as_tensor_variable(y))[0]


class TermSum(Term):
    __doc__ = base_terms.TermSum.__doc__

    def __init__(self, *terms, **kwargs):
        if any(isinstance(term, TermConvolution) for term in terms):
            raise TypeError(
                "You cannot perform operations on an "
                "TermConvolution, it must be the outer term in "
                "the kernel"
            )
        self._terms = terms
        super().__init__(**kwargs)

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.coefficients for t in self.terms)
        return tuple(tt.concatenate(a, axis=0) for a in zip(*coeffs))


class TermProduct(Term):
    __doc__ = base_terms.TermProduct.__doc__

    def __init__(self, term1, term2, **kwargs):
        int1 = isinstance(term1, TermConvolution)
        int2 = isinstance(term2, TermConvolution)
        if int1 or int2:
            raise TypeError(
                "You cannot perform operations on an "
                "TermConvolution, it must be the outer term in "
                "the kernel"
            )
        self.term1 = term1
        self.term2 = term2
        super().__init__(**kwargs)

    def get_coefficients(self):
        c1 = self.term1.coefficients
        c2 = self.term2.coefficients

        # First compute real terms
        ar = []
        cr = []
        ar.append(tt.flatten(c1[0][:, None] * c2[0][None, :]))
        cr.append(tt.flatten(c1[1][:, None] + c2[1][None, :]))

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        ac.append(tt.flatten(c1[0][:, None] * c2[2][None, :]))
        bc.append(tt.flatten(c1[0][:, None] * c2[3][None, :]))
        cc.append(tt.flatten(c1[1][:, None] + c2[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c1[1])[:, None] + c2[5][None, :]))

        ac.append(tt.flatten(c2[0][:, None] * c1[2][None, :]))
        bc.append(tt.flatten(c2[0][:, None] * c1[3][None, :]))
        cc.append(tt.flatten(c2[1][:, None] + c1[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c2[1])[:, None] + c1[5][None, :]))

        # complex * complex
        aj, bj, cj, dj = c1[2:]
        ak, bk, ck, dk = c2[2:]

        ac.append(
            tt.flatten(
                0.5 * (aj[:, None] * ak[None, :] + bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            tt.flatten(
                0.5 * (bj[:, None] * ak[None, :] - aj[:, None] * bk[None, :])
            )
        )
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] - dk[None, :]))

        ac.append(
            tt.flatten(
                0.5 * (aj[:, None] * ak[None, :] - bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            tt.flatten(
                0.5 * (bj[:, None] * ak[None, :] + aj[:, None] * bk[None, :])
            )
        )
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] + dk[None, :]))

        return [
            tt.concatenate(vals, axis=0)
            if len(vals)
            else tt.zeros(0, dtype=self.dtype)
            for vals in (ar, cr, ac, bc, cc, dc)
        ]


class TermDiff(Term):
    __doc__ = base_terms.TermDiff.__doc__

    def __init__(self, term, **kwargs):
        if isinstance(term, TermConvolution):
            raise TypeError(
                "You cannot perform operations on an "
                "TermConvolution, it must be the outer term in "
                "the kernel"
            )
        self.term = term
        super().__init__(**kwargs)

    def get_coefficients(self):
        coeffs = self.term.coefficients
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

    def __init__(self, term, delta, **kwargs):
        self.term = term
        self.delta = tt.as_tensor_variable(delta).astype("float64")
        super().__init__(**kwargs)

    def get_celerite_matrices(self, x, diag):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Real part
        cd = cr * dt
        delta_diag = 2 * tt.sum(ar * (cd - tt.sinh(cd)) / cd ** 2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = tt.sinh(cd)
        cosh = tt.cosh(cd)
        delta_diag += 2 * tt.sum(
            (
                C2 * cosh * tt.sin(dd)
                - C1 * sinh * tt.cos(dd)
                + (a * c + b * d) * dt * c2pd2
            )
            / norm
        )

        new_diag = diag + delta_diag
        return super().get_celerite_matrices(x, new_diag)

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.coefficients

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (tt.cosh(crd) - 1) / crd ** 2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c ** 2
        d2 = d ** 2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = tt.cosh(cd) * tt.cos(dd) - 1
        sin_term = tt.sinh(cd) * tt.sin(dd)

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
        sinc = tt.switch(tt.neq(arg, 0), tt.sin(arg) / arg, tt.ones_like(arg))
        return psd0 * sinc ** 2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Format the lags correctly
        tau0 = tt.abs_(tau0)
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = tt.cosh(crd)
        norm = 2 * ar / crd ** 2
        K_large = tt.sum(norm * (cosh - 1) * tt.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + tt.sum(norm * (crdmt - tt.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = tt.exp(-c * tau)
        cdt = tt.cos(d * tau)
        sdt = tt.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (tt.cosh(cd) * tt.cos(dd) - 1)
        sin_term = 2 * (tt.sinh(cd) * tt.sin(dd))
        factor = k0 * norm
        K_large += tt.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += tt.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = tt.exp(-c * dmt)
        edpt = tt.exp(-c * dpt)
        cos_term = (
            edmt * tt.cos(d * dmt) + edpt * tt.cos(d * dpt) - 2 * k0 * cdt
        )
        sin_term = (
            edmt * tt.sin(d * dmt) + edpt * tt.sin(d * dpt) - 2 * k0 * sdt
        )
        K_small += tt.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += tt.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        return tt.switch(tt.le(tau0, dt), K_small, K_large)


class RealTerm(Term):
    __doc__ = base_terms.RealTerm.__doc__

    def __init__(self, *, a, c, **kwargs):
        self.a = tt.as_tensor_variable(a).astype("float64")
        self.c = tt.as_tensor_variable(c).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = tt.zeros(0, dtype=self.dtype)
        if self.a.ndim == 0:
            return (
                self.a[None],
                self.c[None],
                empty,
                empty,
                empty,
                empty,
            )
        return (self.a, self.c, empty, empty, empty, empty)


class ComplexTerm(Term):
    __doc__ = base_terms.ComplexTerm.__doc__

    def __init__(self, *, a, b, c, d, **kwargs):
        self.a = tt.as_tensor_variable(a).astype("float64")
        self.b = tt.as_tensor_variable(b).astype("float64")
        self.c = tt.as_tensor_variable(c).astype("float64")
        self.d = tt.as_tensor_variable(d).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = tt.zeros(0, dtype=self.dtype)
        if self.a.ndim == 0:
            return (
                empty,
                empty,
                self.a[None],
                self.b[None],
                self.c[None],
                self.d[None],
            )
        return (empty, empty, self.a, self.b, self.c, self.d)


class SHOTerm(Term):
    __doc__ = base_terms.SHOTerm.__doc__
    __parameter_spec__ = base_terms.SHOTerm.__parameter_spec__

    @base_terms.handle_parameter_spec(
        lambda x: tt.as_tensor_variable(x).astype("float64")
    )
    def __init__(self, *, eps=1e-5, **kwargs):
        self.eps = tt.as_tensor_variable(eps).astype("float64")
        super().__init__(**kwargs)

    def overdamped(self):
        Q = self.Q
        f = tt.sqrt(tt.maximum(1.0 - 4.0 * Q ** 2, self.eps))
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * tt.stack([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * tt.stack([1.0 - f, 1.0 + f]),
            empty,
            empty,
            empty,
            empty,
        )

    def underdamped(self):
        Q = self.Q
        f = tt.sqrt(tt.maximum(4.0 * Q ** 2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            tt.stack([a]),
            tt.stack([a / f]),
            tt.stack([c]),
            tt.stack([c * f]),
        )

    def get_coefficients(self):
        m = self.Q < 0.5
        return [
            ifelse(m, a, b)
            for a, b in zip(self.overdamped(), self.underdamped())
        ]


class Matern32Term(Term):
    __doc__ = base_terms.Matern32Term.__doc__

    def __init__(self, *, sigma, rho, eps=0.01, **kwargs):
        self.sigma = tt.as_tensor_variable(sigma).astype("float64")
        self.rho = tt.as_tensor_variable(rho).astype("float64")
        self.eps = tt.as_tensor_variable(eps).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        w0 = np.sqrt(3.0) / self.rho
        S0 = self.sigma ** 2 / w0
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            (w0 * S0)[None],
            (w0 * w0 * S0 / self.eps)[None],
            w0[None],
            self.eps[None],
        )


class RotationTerm(TermSum):
    __doc__ = base_terms.RotationTerm.__doc__

    def __init__(self, *, sigma, period, Q0, dQ, f, **kwargs):
        self.sigma = tt.as_tensor_variable(sigma).astype("float64")
        self.period = tt.as_tensor_variable(period).astype("float64")
        self.Q0 = tt.as_tensor_variable(Q0).astype("float64")
        self.dQ = tt.as_tensor_variable(dQ).astype("float64")
        self.f = tt.as_tensor_variable(f).astype("float64")

        self.amp = self.sigma ** 2 / (1 + self.f)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * tt.sqrt(4 * Q1 ** 2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * tt.sqrt(4 * Q2 ** 2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2), **kwargs
        )
