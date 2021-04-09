# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
    # "TermDiff",
    # "TermConvolution",
    "CeleriteTerm",
    "SHOTerm",
    "RotationTerm",
    "Matern32Term",
]

from collections import namedtuple
from functools import wraps

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from . import ops

CeleriteSystem = namedtuple("CeleriteSystem", ["a", "U", "V", "P"])


class Term:
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload at least the :func:`Term.get_celerite_matrices`
    method.

    """

    def __add__(self, other):
        return TermSum(self, other)

    def get_celerite_matrices(self, x, diag=None):
        """Get the matrices needed to solve the celerite system

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N], optional): The diagonal variance of the system.
        """
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """
        raise NotImplementedError("subclasses must implement this method")

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        raise NotImplementedError("subclasses must implement this method")

    def dot(self, x, diag, y):
        """Apply a matrix-vector or matrix-matrix product

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N]): The diagonal variance of the system.
            y (shape[N] or shape[N, K]): The target of vector or matrix for
                this operation.
        """
        is_vector = False
        y = jnp.atleast_1d(y)
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]

        a, U, V, P = self.get_celerite_matrices(x, diag=diag)
        res = (
            a[:, None] * y
            + ops.matmul_upper(U, V, P, y)
            + ops.matmul_lower(U, V, P, y)
        )
        if is_vector:
            return res[:, 0]
        return res


class TermSum(Term):
    def __init__(self, *terms):
        if not len(terms):
            raise ValueError("A TermSum must include at least one term")
        self.terms = terms

    def get_celerite_matrices(self, x, diag=None):
        # Helper function to handle general Ps
        def combine_p(P1, P2):
            if P1.ndim == P2.ndim:
                return jnp.concatenate((P1, P2), axis=-1)
            if P1.ndim < P2.ndim:
                P1 = jax.vmap(jnp.diag)(P1)
            else:
                P2 = jax.vmap(jnp.diag)(P2)
            return jax.vmap(jsp.linalg.block_diag)(P1, P2)

        a, U, V, P = self.terms[0].get_celerite_matrices(x, diag=diag)
        for term in self.terms[1:]:
            a, Up, Vp, Pp = term.get_celerite_matrices(x, diag=a)
            U = jnp.concatenate((U, Up), axis=-1)
            V = jnp.concatenate((V, Vp), axis=-1)
            P = combine_p(P, Pp)
        return CeleriteSystem(a, U, V, P)

    def get_value(self, tau):
        k = self.terms[0].get_value(tau)
        for term in self.terms[1:]:
            k += term.get_value(tau)
        return k

    def get_psd(self, omega):
        p = self.terms[0].get_psd(omega)
        for term in self.terms[1:]:
            p += term.get_psd(omega)
        return p


class CeleriteTerm(Term):
    r"""A general celerite term

    .. warning:: You should only use this term if you really know what you're
        doing because it will generally behave poorly. Check out
        :class:`SHOTerm` instead!
    """

    def __init__(self, *, a, b, c, d):
        self.a, self.b, self.c, self.d = jnp.broadcast_arrays(
            *map(jnp.atleast_1d, (a, b, c, d))
        )
        self.is_real = self.b < 0
        self.abs_b = jnp.sqrt(jnp.abs(self.b))

    def __add__(self, other):
        if isinstance(other, CeleriteTerm):
            return CeleriteTerm(
                a=jnp.concatenate((self.a, other.a)),
                b=jnp.concatenate((self.b, other.b)),
                c=jnp.concatenate((self.c, other.c)),
                d=jnp.concatenate((self.d, other.d)),
            )
        return TermSum(self, other)

    def get_celerite_matrices(self, x, diag=None):
        x = jnp.asarray(x)
        if diag is None:
            diag = jnp.zeros_like(x)
        else:
            diag = jnp.asarray(diag)
            if diag.shape != x.shape:
                raise ValueError("dimension mismatch")

        dtype = x.dtype
        shape = tuple(x.shape) + tuple(self.a.shape)

        t = x[..., None]
        arg = self.d * t
        sin = jnp.sin(arg)
        cos = jnp.cos(arg)

        U1 = jnp.where(
            self.is_real,
            jnp.broadcast_to(0.5 * self.a * (1 + self.abs_b), shape),
            self.a * (cos + self.abs_b * sin),
        )
        U2 = jnp.where(
            self.is_real,
            jnp.broadcast_to(0.5 * self.a * (1 - self.abs_b), shape),
            self.a * (sin - self.abs_b * cos),
        )

        V1 = jnp.where(self.is_real, jnp.ones(shape, dtype=dtype), cos)
        V2 = jnp.where(self.is_real, jnp.ones(shape, dtype=dtype), sin)

        dx = jnp.append(0, jnp.diff(x))[..., None]
        arg = jnp.exp(-self.c * dx)
        P1 = jnp.where(self.is_real, jnp.exp(-(self.c - self.d) * dx), arg)
        P2 = jnp.where(self.is_real, jnp.exp(-(self.c + self.d) * dx), arg)

        return CeleriteSystem(
            diag + jnp.sum(self.a),
            jnp.concatenate((U1, U2), axis=-1),
            jnp.concatenate((V1, V2), axis=-1),
            jnp.concatenate((P1, P2), axis=-1),
        )

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """

        def real(tau):
            ar1 = 0.5 * self.a * (1 + self.abs_b)
            cr1 = self.c - self.d
            ar2 = 0.5 * self.a * (1 - self.abs_b)
            cr2 = self.c + self.d
            return ar1 * jnp.exp(-cr1 * tau) + ar2 * jnp.exp(-cr2 * tau)

        def comp(tau):
            arg = self.d * tau
            return (
                self.a
                * jnp.exp(-self.c * tau)
                * (jnp.cos(arg) + self.abs_b * jnp.sin(arg))
            )

        tau = jnp.abs(tau)[..., None]
        return jnp.sum(jnp.where(self.is_real, real(tau), comp(tau)), axis=-1)

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """

        def real(w2):
            ar1 = 0.5 * self.a * (1 + self.abs_b)
            cr1 = self.c - self.d
            ar2 = 0.5 * self.a * (1 - self.abs_b)
            cr2 = self.c + self.d
            return ar1 * cr1 / (jnp.square(cr1) + w2) + ar2 * cr2 / (
                jnp.square(cr2) + w2
            )

        def comp(w2):
            c2 = jnp.square(self.c)
            d2 = jnp.square(self.d)
            w02 = c2 + d2
            return (
                self.a
                * (
                    (self.c + self.abs_b * self.d) * w02
                    + (self.c - self.abs_b * self.d) * w2
                )
                / (jnp.square(w2) + 2 * (c2 - d2) * w2 + w02 * w02)
            )

        w2 = jnp.square(omega)[..., None]
        return np.sqrt(2 / np.pi) * jnp.sum(
            jnp.where(self.is_real, real(w2), comp(w2)), axis=-1
        )


def handle_parameter_spec(to_wrap):
    @wraps(to_wrap)
    def wrapped(target, *args, **kwargs):
        for param, alt in target.__parameter_spec__:
            all_names = set([param]) | set(name for name, _ in alt)
            if sum(int(n in kwargs) for n in all_names) != 1:
                raise ValueError(
                    "exactly one of {0} must be defined".format(
                        list(all_names)
                    )
                )
            if param in kwargs:
                setattr(target, param, kwargs.pop(param))
            else:
                for name, func in alt:
                    if name in kwargs:
                        setattr(
                            target,
                            param,
                            func(target, kwargs.pop(name)),
                        )
                        break

        return to_wrap(target, *args, **kwargs)

    return wrapped


class SHOTerm(CeleriteTerm):
    r"""A term representing a stochastically-driven, damped harmonic oscillator

    The PSD of this term is

    .. math::

        S(\omega) = \sqrt{\frac{2}{\pi}} \frac{S_0\,\omega_0^4}
        {(\omega^2-{\omega_0}^2)^2 + {\omega_0}^2\,\omega^2/Q^2}

    with the parameters ``S0``, ``Q``, and ``w0``.

    This implementation also supports the following reparameterizations that
    can be easier to use and reason about:

    1. ``rho``, the undamped period of the oscillator, defined as :math:`\rho
       = 2\,\pi / \omega_0`,
    2. ``tau``, the damping timescale of the process, defined as :math:`\tau =
       2\,Q / \omega_0`, and
    3. ``sigma``, the standard deviation of the process, defined as
       :math:`\sigma = \sqrt{S_0\,\omega_0\,Q}`.

    Args:
        w0: The undamped angular frequency, :math:`\omega_0` above.
        rho: Alternative parameterization for ``w0`` as described above.
        Q: The quality factor, :math:`Q` above.
        tau: Alternative parameterization for ``Q`` as described above.
        S0: The power at :math:`\omega = 0`, :math:`S_0` above.
        sigma: Alternative parameterization for ``S0`` as described above.
        eps (optional): A regularization parameter used for numerical stability
            when computing :math:`\sqrt{1-4\,Q^2}` or :math:`\sqrt{4\,Q^2-1}`.
    """

    __parameter_spec__ = (
        ("w0", (("rho", lambda self, rho: 2 * np.pi / rho),)),
        ("Q", (("tau", lambda self, tau: 0.5 * self.w0 * tau),)),
        (
            "S0",
            (("sigma", lambda self, sigma: sigma ** 2 / (self.w0 * self.Q)),),
        ),
    )

    @handle_parameter_spec
    def __init__(self):
        a, b, c, d = SHOTerm.get_parameters(
            *jnp.broadcast_arrays(
                *map(jnp.atleast_1d, (self.S0, self.w0, self.Q))
            )
        )
        super().__init__(a=a, b=b, c=c, d=d)

    @staticmethod
    def get_parameters(S0, w0, Q):
        f2 = 4.0 * jnp.square(Q) - 1.0
        a = S0 * w0 * Q
        c = 0.5 * w0 / Q
        return a, 1 / f2, c, c * jnp.sqrt(jnp.abs(f2))


class RotationTerm(CeleriteTerm):
    r"""A mixture of two SHO terms that can be used to model stellar rotation

    This term has two modes in Fourier space: one at ``period`` and one at
    ``0.5 * period``. This can be a good descriptive model for a wide range of
    stochastic variability in stellar time series from rotation to pulsations.

    More precisely, the parameters of the two :class:`SHOTerm` terms are
    defined as

    .. math::

        Q_1 = 1/2 + Q_0 + \delta Q \\
        \omega_1 = \frac{4\,\pi\,Q_1}{P\,\sqrt{4\,Q_1^2 - 1}} \\
        S_1 = \frac{\sigma^2}{(1 + f)\,\omega_1\,Q_1}

    for the primary term, and

    .. math::

        Q_2 = 1/2 + Q_0 \\
        \omega_2 = \frac{8\,\pi\,Q_1}{P\,\sqrt{4\,Q_1^2 - 1}} \\
        S_2 = \frac{f\,\sigma^2}{(1 + f)\,\omega_2\,Q_2}

    for the secondary term.

    Args:
        sigma: The standard deviation of the process.
        period: The primary period of variability.
        Q0: The quality factor (or really the quality factor minus one half;
            this keeps the system underdamped) for the secondary oscillation.
        dQ: The difference between the quality factors of the first and the
            second modes. This parameterization (if ``dQ > 0``) ensures that
            the primary mode alway has higher quality.
        f: The fractional amplitude of the secondary mode compared to the
            primary. This should probably always be ``0 < f < 1``, but that
            is not enforced.
    """

    def __init__(self, *, sigma, period, Q0, dQ, f):
        sigma, period, Q0, dQ, f = jnp.broadcast_arrays(
            *map(jnp.atleast_1d, (sigma, period, Q0, dQ, f))
        )

        amp = jnp.square(sigma) / (1 + f)

        # One term with a period of period
        Q1 = 0.5 + Q0 + dQ
        w1 = 4 * np.pi * Q1 / (period * jnp.sqrt(4 * jnp.square(Q1) - 1))
        S1 = amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + Q0
        w2 = 8 * np.pi * Q2 / (period * jnp.sqrt(4 * jnp.square(Q2) - 1))
        S2 = f * amp / (w2 * Q2)

        a1, b1, c1, d1 = SHOTerm.get_parameters(S1, w1, Q1)
        a2, b2, c2, d2 = SHOTerm.get_parameters(S2, w2, Q2)

        super().__init__(
            a=jnp.concatenate((a1, a2)),
            b=jnp.concatenate((b1, b2)),
            c=jnp.concatenate((c1, c2)),
            d=jnp.concatenate((d1, d2)),
        )


class KalmanTerm(Term):
    def __init__(self, *, sigma, h):
        self.s2 = jnp.square(sigma)
        self.h = h

    def phi(self, dx):
        raise NotImplementedError("sublasses must implement this method")

    def get_celerite_matrices(self, x, diag=None):
        x = jnp.asarray(x)
        if diag is None:
            diag = jnp.zeros_like(x)
        else:
            diag = jnp.asarray(diag)
            if diag.shape != x.shape:
                raise ValueError("dimension mismatch")

        dx = jnp.append(0, jnp.diff(x))
        P = self.phi(dx)
        V = jnp.repeat(self.h[None], x.shape[0], axis=0)
        return CeleriteSystem(self.s2 + diag, self.s2 * V, V, P)


class Matern32Term(KalmanTerm):
    r"""A Matern-3/2 term

    .. math::

        k(\tau) = \sigma^2\,\left(1+\frac{\sqrt{3}\,\tau}{\rho}\right)\,
        \exp\left(-\frac{\sqrt{3}\,\tau}{\rho}\right)

    with the parameters ``sigma`` and ``rho``.

    Args:
        sigma: The parameter :math:`\sigma`.
        rho: The parameter :math:`\rho`.
    """

    def __init__(self, *, sigma, rho):
        f = self.f = np.sqrt(3) / rho
        self.phi1 = jnp.array([[f, 1.0], [-jnp.square(f), -f]])[None]
        super().__init__(sigma=sigma, h=jnp.array([1.0, 0.0]))

    def phi(self, dx):
        return jnp.exp(-self.f * dx)[:, None, None] * (
            jnp.eye(2)[None] + dx[:, None, None] * self.phi1
        )

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """
        tau = self.f * jnp.abs(tau)
        return self.s2 * (1 + tau) * jnp.exp(-tau)

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        w2 = jnp.square(omega)
        return (
            2
            * np.sqrt(2 / np.pi)
            * self.s2
            * self.f ** 3
            / (w2 + self.f ** 2) ** 2
        )
