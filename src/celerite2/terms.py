# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    # "TermSum",
    # "TermProduct",
    # "TermDiff",
    # "TermConvolution",
    # "RealTerm",
    # "ComplexTerm",
    # "SHOTerm",
    # "Matern32Term",
    # "RotationTerm",
    # "OriginalCeleriteTerm",
]

from functools import wraps

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


class Term:
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload at least the :func:`Term.get_celerite_matrices`
    method.

    """

    def __add__(self, b):
        return TermSum(self, b)

    @property
    def terms(self):
        return [self]

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


class CeleriteTerm(Term):
    r"""A general celerite term

    .. warning:: You should only use this term if you really know what you're
        doing because it will generally behave poorly. Check out
        :class:`SHOTerm` instead!
    """

    def __init__(self, *, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.is_real = self.b <= 0
        self.abs_b = jnp.sqrt(jnp.abs(self.b))

    def get_celerite_matrices(self, x, diag=None):
        x = jnp.asarray(x)
        if diag is None:
            diag = jnp.zeros_like(x)
        else:
            diag = jnp.asarray(diag)
            if diag.shape != x.shape:
                raise ValueError("dimension mismatch")

        U1 = lax.cond(
            self.is_real,
            lambda x_: jnp.broadcast_to(
                0.5 * self.a * (1 + self.abs_b), x_.shape
            ),
            lambda x_: self.a
            * (jnp.cos(self.d * x_) + self.abs_b * jnp.sin(self.d * x_)),
            operand=x,
        )
        U2 = lax.cond(
            self.is_real,
            lambda x_: jnp.broadcast_to(
                0.5 * self.a * (1 - self.abs_b), x_.shape
            ),
            lambda x_: self.a
            * (jnp.sin(self.d * x_) - self.abs_b * jnp.cos(self.d * x_)),
            operand=x,
        )

        V1 = lax.cond(
            self.is_real,
            lambda x_: jnp.ones_like(x_),
            lambda x_: jnp.cos(self.d * x_),
            operand=x,
        )
        V2 = lax.cond(
            self.is_real,
            lambda x_: jnp.ones_like(x_),
            lambda x_: jnp.sin(self.d * x_),
            operand=x,
        )

        dx = jnp.append(0, jnp.diff(x))
        P1 = lax.cond(
            self.is_real,
            lambda dx_: jnp.exp(-(self.c - self.d) * dx_),
            lambda dx_: jnp.exp(-self.c * dx_),
            operand=dx,
        )
        P2 = lax.cond(
            self.is_real,
            lambda dx_: jnp.exp(-(self.c + self.d) * dx_),
            lambda dx_: jnp.exp(-self.c * dx_),
            operand=dx,
        )

        return (
            diag + self.a,
            jnp.stack((U1, U2), axis=-1),
            jnp.stack((V1, V2), axis=-1),
            jnp.stack((P1, P2), axis=-1),
        )

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """

        def real(tau):
            ar1 = self.a * (1 + self.abs_b)
            cr1 = self.c - self.d
            ar2 = self.a * (1 - self.abs_b)
            cr2 = self.c + self.d
            return ar1 * jnp.exp(-cr1 * tau) + ar2 * jnp.exp(-cr2 * tau)

        def comp(tau):
            arg = self.d * tau
            return (
                self.a
                * jnp.exp(-self.c * tau)
                * (jnp.cos(arg) + self.abs_b * jnp.sin(arg))
            )

        return lax.cond(self.is_real, real, comp, operand=jnp.abs(tau))

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """

        def real(w2):
            ar1 = self.a + self.abs_b
            cr1 = self.c - self.d
            ar2 = self.a - self.abs_b
            cr2 = self.c + self.d
            return ar1 * cr1 / (jnp.square(cr1) + w2) + ar2 * cr2 / (
                jnp.square(cr2) + w2
            )

        def comp(w2):
            c2 = jnp.square(self.c)
            d2 = jnp.square(self.d)
            w02 = c2 + d2
            return (
                (self.a * self.c + self.abs_b * self.d) * w02
                + (self.a * self.c - self.abs_b * self.d) * w2
            ) / (jnp.square(w2) + 2 * (c2 - d2) * w2 + w02 * w02)

        return np.sqrt(2 / np.pi) * lax.cond(
            self.is_real, real, comp, operand=jnp.square(omega)
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
        f2 = 4.0 * jnp.square(self.Q) - 1.0
        a = self.S0 * self.w0 * self.Q
        c = 0.5 * self.w0 / self.Q
        super().__init__(a=a, b=1 / f2, c=c, d=c * jnp.sqrt(jnp.abs(f2)))


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

        # V = jnp.stack((jnp.ones_like(x), jnp.zeros_like(x)), axis=-1)
        dx = jnp.append(0, jnp.diff(x))
        P = self.phi(dx)
        V = jnp.repeat(self.h[None], x.shape[0], axis=0)
        return self.s2 + diag, self.s2 * V, V, P


class Matern32Term(KalmanTerm):
    r"""A Matern-3/2 term

    .. math::

        k(\tau) = \sigma^2\,\left(1+ \frac{\sqrt{3}\,\tau}{\rho}\right)\,
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


class Matern52Term(KalmanTerm):
    def __init__(self, *, sigma, rho):
        f = self.f = np.sqrt(5) / rho
        f2 = jnp.square(f)
        f3 = f * f2
        self.phi1 = jnp.array(
            [[f, 1.0, 0.0], [0.0, f, 1.0], [-f3, -3 * f2, -2 * f]]
        )[None]
        self.phi2 = 0.5 * jnp.array(
            [
                [f2, 2 * f, 1.0],
                [-f3, -2 * f2, -f],
                [f2 * f2, 2 * f3, f2],
            ]
        )[None]
        super().__init__(sigma=sigma, h=jnp.array([1.0, 0.0, 0.0]))

    def phi(self, dx):
        return jnp.exp(-self.f * dx)[:, None, None] * (
            jnp.eye(3)[None]
            + dx[:, None, None] * self.phi1
            + jnp.square(dx)[:, None, None] * self.phi2
        )
