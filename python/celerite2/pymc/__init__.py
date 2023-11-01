# -*- coding: utf-8 -*-

__all__ = ["terms", "GaussianProcess"]


def __set_compiler_flags():
    import pytensor

    def add_flag(current, new):
        if new in current:
            return current
        return f"{current} {new}"

    current = pytensor.config.gcc__cxxflags
    current = add_flag(current, "-Wno-c++11-narrowing")
    current = add_flag(current, "-fno-exceptions")
    current = add_flag(current, "-fno-unwind-tables")
    current = add_flag(current, "-fno-asynchronous-unwind-tables")
    pytensor.config.gcc__cxxflags = current


__set_compiler_flags()

from celerite2.pymc import terms  # noqa
from celerite2.pymc.celerite2 import GaussianProcess  # noqa

try:
    from celerite2.pymc import jax_support  # noqa
except ImportError:
    pass
