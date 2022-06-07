# -*- coding: utf-8 -*-

__all__ = ["terms", "GaussianProcess"]


def __set_compiler_flags():
    import theano

    def add_flag(current, new):
        if new in current:
            return current
        return f"{current} {new}"

    current = theano.config.gcc__cxxflags
    current = add_flag(current, "-Wno-c++11-narrowing")
    current = add_flag(current, "-fno-exceptions")
    current = add_flag(current, "-fno-unwind-tables")
    current = add_flag(current, "-fno-asynchronous-unwind-tables")
    theano.config.gcc__cxxflags = current


__set_compiler_flags()

from celerite2.pymc3 import terms
from celerite2.pymc3.celerite2 import GaussianProcess
