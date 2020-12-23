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
    __requires_general_addition__ = False

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

    def get_value(self, t, t2=None, *, diag=None, X=None, X2=None):
        t = tt.as_tensor_variable(t)
        if diag is None:
            diag = tt.zeros_like(t)
        c1, a1, U1, V1 = self.get_celerite_matrices(t, diag, X=X)
        if t2 is None:
            Y = tt.eye(t.shape[0])
            K = tt.diag(a1)
            K += ops.matmul_lower(t, c1, U1, V1, Y)[0]
            K += ops.matmul_upper(t, c1, U1, V1, Y)[0]
            return K

        t2 = tt.as_tensor_variable(t2)
        c2, a2, U2, V2 = self.get_celerite_matrices(
            t2, tt.zeros_like(t2), X=X2
        )
        Y = tt.eye(t2.shape[0])
        K = tt.zeros((t.shape[0], t2.shape[0]))
        K += ops.general_matmul_lower(t, t2, c1, U1, V2, Y)[0]
        K += ops.general_matmul_upper(t, t2, c1, V1, U2, Y)[0]
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

    def get_celerite_matrices(self, x, diag, **kwargs):
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

        c = tt.concatenate((cr, cc, cc))

        return c, a, U, V

    def dot(self, x, diag, y, *, X=None):
        x = tt.as_tensor_variable(x)
        y = tt.as_tensor_variable(y)

        is_vector = False
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("'y' can only be a vector or matrix")

        c, a, U, V = self.get_celerite_matrices(x, diag, X=X)
        z = y * a[:, None]
        z += ops.matmul_lower(x, c, U, V, y)[0]
        z += ops.matmul_upper(x, c, U, V, y)[0]

        if is_vector:
            return z[:, 0]
        return z


class TermSum(Term):
    __doc__ = base_terms.TermSum.__doc__

    def __init__(self, *terms, **kwargs):
        if any(term.__requires_general_addition__ for term in terms):
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
            )
        self._terms = terms
        super().__init__(**kwargs)

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.coefficients for t in self.terms)
        return tuple(tt.concatenate(a, axis=0) for a in zip(*coeffs))


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
        diag = tt.as_tensor_variable(diag)
        matrices = [
            term.get_celerite_matrices(t, tt.zeros_like(diag), X=X)
            for term in self.terms
        ]
        c = tt.concatenate([m[0] for m in matrices])
        a = diag + tt.sum([m[1] for m in matrices], axis=0)
        U = tt.concatenate([m[2] for m in matrices], axis=1)
        V = tt.concatenate([m[3] for m in matrices], axis=1)
        return c, a, U, V


class TermProduct(Term):
    __doc__ = base_terms.TermProduct.__doc__

    def __init__(self, term1, term2, **kwargs):
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
        if term.__requires_general_addition__:
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
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
    __requires_general_addition__ = True

    def __init__(self, term, delta, **kwargs):
        if term.__requires_general_addition__:
            raise TypeError(
                "You cannot perform operations on a term that requires general"
                " addition, it must be the outer term in the kernel"
            )
        self.term = term
        self.delta = tt.as_tensor_variable(delta).astype("float64")
        super().__init__(**kwargs)

    def get_celerite_matrices(self, x, diag, **kwargs):
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
