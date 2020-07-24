# -*- coding: utf-8 -*-

__all__ = [
    "factor_fwd",
    "factor_rev",
    "solve_fwd",
    "solve_rev",
    "norm_fwd",
    "norm_rev",
    "dot_tril_fwd",
    "dot_tril_rev",
    "matmul_fwd",
    "matmul_rev",
]
import numpy as np

from celerite2 import backprop


def factor_fwd(a, U, V, P):
    N, J = U.shape
    d = np.empty_like(a)
    W = np.empty_like(V)
    S = np.empty((N, J ** 2), dtype=np.float64)
    return backprop.factor_fwd(a, U, V, P, d, W, S)


def factor_rev(a, U, V, P, d, W, S, bd, bW):
    ba = np.empty_like(a)
    bU = np.empty_like(U)
    bV = np.empty_like(V)
    bP = np.empty_like(P)
    return backprop.factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP)


def solve_fwd(U, P, d, W, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    Z = np.empty_like(X)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    G = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.solve_fwd(U, P, d, W, Y, X, Z, F, G)


def solve_rev(U, P, d, W, Y, X, Z, F, G, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.solve_rev(
        U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY
    )


def norm_fwd(U, P, d, W, Y):
    N, J = U.shape
    X = np.empty((), dtype=np.float64)
    Z = np.empty_like(Y)
    F = np.empty((N, J), dtype=np.float64)
    return backprop.norm_fwd(U, P, d, W, Y, X, Z, F)


def norm_rev(U, P, d, W, Y, X, Z, F, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.norm_rev(U, P, d, W, Y, X, Z, F, bX, bU, bP, bd, bW, bY)


def dot_tril_fwd(U, P, d, W, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.dot_tril_fwd(U, P, d, W, Y, X, F)


def dot_tril_rev(U, P, d, W, Y, X, F, bX):
    bU = np.empty_like(U)
    bP = np.empty_like(P)
    bd = np.empty_like(d)
    bW = np.empty_like(W)
    bY = np.empty_like(Y)
    return backprop.dot_tril_rev(U, P, d, W, Y, X, F, bX, bU, bP, bd, bW, bY)


def matmul_fwd(a, U, V, P, Y):
    N, J = U.shape
    if Y.ndim == 1:
        nrhs = 1
    else:
        nrhs = Y.shape[1]
    X = np.empty_like(Y)
    Z = np.empty_like(X)
    F = np.empty((N, J * nrhs), dtype=np.float64)
    G = np.empty((N, J * nrhs), dtype=np.float64)
    return backprop.matmul_fwd(a, U, V, P, Y, X, Z, F, G)


def matmul_rev(a, U, V, P, Y, X, Z, F, G, bX):
    ba = np.empty_like(a)
    bU = np.empty_like(U)
    bV = np.empty_like(V)
    bP = np.empty_like(P)
    bY = np.empty_like(Y)
    return backprop.matmul_rev(
        a, U, V, P, Y, X, Z, F, G, bX, ba, bU, bV, bP, bY
    )
