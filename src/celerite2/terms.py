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

import jax.numpy as jnp
import numpy as np


class Term:
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload at least the :func:`Term.get_coefficients`
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
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        raise NotImplementedError("subclasses must implement this method")


class CeleriteTerm(Term):
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

        arg = self.d * x
        cos = jnp.cos(arg)
        sin = jnp.sin(arg)

        U1 = jnp.where(
            self.is_real,
            jnp.broadcast_to(self.a + self.abs_b, x.shape),
            self.a * cos + self.abs_b * sin,
        )
        U2 = jnp.where(
            self.is_real,
            jnp.broadcast_to(self.a - self.abs_b, x.shape),
            self.a * sin - self.abs_b * cos,
        )

        V1 = jnp.where(self.is_real, jnp.ones_like(x), cos)
        V2 = jnp.where(self.is_real, jnp.ones_like(x), sin)

        dx = jnp.append(0, jnp.diff(x))
        exp = jnp.exp(-self.c * dx)
        P1 = jnp.where(self.is_real, jnp.exp(-(self.c + self.d) * dx), exp)
        P2 = jnp.where(self.is_real, jnp.exp(-(self.c - self.d) * dx), exp)

        return (
            diag + self.a,
            jnp.stack((U1, U2), axis=-1),
            jnp.stack((V1, V2), axis=-1),
            jnp.stack((P1, P2), axis=-1),
        )

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        w2 = jnp.asarray(omega) ** 2

        ar1 = self.a + self.abs_b
        cr1 = self.c + self.d
        ar2 = self.a - self.abs_b
        cr2 = self.c - self.d
        real = ar1 * cr1 / (cr1 ** 2 + w2) + ar2 * cr2 / (cr2 ** 2 + w2)

        c2 = jnp.square(self.c)
        d2 = jnp.square(self.d)
        w02 = c2 + d2
        comp = (
            (self.a * self.c + self.abs_b * self.d) * w02
            + (self.a * self.c - self.abs_b * self.d) * w2
        ) / (w2 ** 2 + 2.0 * (c2 - d2) * w2 + w02 * w02)

        return np.sqrt(2 / np.pi) * jnp.where(self.is_real, real, comp)


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
        super().__init__(
            a=a, b=jnp.square(a) / f2, c=c, d=-c * jnp.sqrt(jnp.abs(f2))
        )
