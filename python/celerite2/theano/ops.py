# -*- coding: utf-8 -*-

__all__ = [
    "factor",
    "factor_quiet",
    "solve",
    "norm",
    "dot_tril",
    "matmul",
    "conditional_mean",
]
from itertools import chain

import numpy as np
import theano
from theano import tensor as tt

from .. import backprop, driver


def _resize_or_set(outputs, n, shape):
    if outputs[n][0] is None:
        outputs[n][0] = np.empty(shape)
    else:
        outputs[n][0] = np.ascontiguousarray(
            np.resize(outputs[n][0], shape), dtype=np.float64
        )
    return outputs[n][0]


class BaseOp(theano.Op):
    __props__ = ()
    idim = ()
    odim = ()
    dim_map = {}
    dtypes = ()

    def make_node(self, *inputs):
        if self.dtypes:
            dtypes = self.dtypes
        else:
            dtypes = ("float64" for _ in self.idim)

        # Check the dtype
        if any(arg.dtype != dtype for arg, dtype in zip(inputs, dtypes)):
            raise NotImplementedError(
                "invalid dtype\n"
                f"expected {dtypes}\n"
                f"got {[arg.dtype for arg in inputs]}"
            )

        # Check the number of inputs
        if len(self.idim) != len(inputs):
            raise ValueError(
                f"expected {len(self.idim)} inputs; got {len(inputs)}"
            )

        # Loop over inputs and check the dimensions of each
        actual_map = {}
        type_map = {}
        for n, ip in enumerate(inputs):
            if ip.ndim == self.idim[n]:
                continue
            if self.idim[n] not in self.dim_map:
                raise ValueError(
                    f"invalid dimensions for input {n} (zero indexed)"
                )
            if self.idim[n] in actual_map:
                if ip.ndim != actual_map[self.idim[n]]:
                    raise ValueError(
                        f"invalid dimensions for input {n} (zero indexed)"
                    )
            if ip.ndim in self.dim_map[self.idim[n]]:
                actual_map[self.idim[n]] = ip.ndim
                type_map[self.idim[n]] = ip.type

        # Build the output types
        otypes = []
        for d in self.odim:
            if d == 0:
                otypes.append(tt.dscalar())
            elif d == 1:
                otypes.append(tt.dvector())
            elif d == 2:
                otypes.append(tt.dmatrix())
            else:
                otypes.append(type_map[d]())

        return theano.Apply(self, inputs, otypes)


class FactorOp(BaseOp):
    __props__ = ("quiet",)
    idim = (1, 2, 2, 2)
    odim = (1, 2, 2)

    def __init__(self, *, quiet=False):
        self.quiet = quiet
        self.rev_op = FactorRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        N = shapes[1][0]
        J = shapes[1][1]
        return shapes[0], shapes[1], [N, J * J]

    def perform(self, node, inputs, outputs):
        a, U, V, P = inputs
        N, J = U.shape

        d = _resize_or_set(outputs, 0, (N,))
        W = _resize_or_set(outputs, 1, (N, J))
        S = _resize_or_set(outputs, 2, (N, J * J))

        try:
            backprop.factor_fwd(a, U, V, P, d, W, S)
        except backprop.LinAlgError:
            if not self.quiet:
                raise
            d[:] = -1.0

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        grads = (
            tt.zeros_like(outputs[n])
            if isinstance(b.type, theano.gradient.DisconnectedType)
            else b
            for n, b in enumerate(gradients[:2])
        )
        return self.rev_op(*chain(inputs, outputs, grads))


class FactorRevOp(BaseOp):
    idim = (1, 2, 2, 2, 1, 2, 2, 1, 2)
    odim = (1, 2, 2, 2)

    def infer_shape(self, node, shapes):
        return shapes[:4]

    def perform(self, node, inputs, outputs):
        a, U, V, P, d, W, S, bd, bW = inputs
        N, J = U.shape

        ba = _resize_or_set(outputs, 0, (N,))
        bU = _resize_or_set(outputs, 1, (N, J))
        bV = _resize_or_set(outputs, 2, (N, J))
        bP = _resize_or_set(outputs, 3, (N - 1, J))

        backprop.factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP)


factor = FactorOp()
factor_quiet = FactorOp(quiet=True)


class SolveOp(BaseOp):
    idim = (2, 2, 1, 2, "rhs")
    odim = ("rhs", "rhs", 2, 2)
    dim_map = {"rhs": (1, 2)}

    def __init__(self):
        self.rev_op = SolveRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        N = shapes[0][0]
        J = shapes[0][1]
        try:
            nrhs = shapes[4][1]
        except IndexError:
            nrhs = 1
        return shapes[4], shapes[4], [N, nrhs * J], [N, nrhs * J]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y = inputs
        N, J = U.shape

        if Y.ndim == 1:
            X = _resize_or_set(outputs, 0, (N,))
            Z = _resize_or_set(outputs, 1, (N,))
            F = _resize_or_set(outputs, 2, (N, J))
            G = _resize_or_set(outputs, 3, (N, J))

        else:
            nrhs = Y.shape[1]
            X = _resize_or_set(outputs, 0, (N, nrhs))
            Z = _resize_or_set(outputs, 1, (N, nrhs))
            F = _resize_or_set(outputs, 2, (N, nrhs * J))
            G = _resize_or_set(outputs, 3, (N, nrhs * J))

        backprop.solve_fwd(U, P, d, W, Y, X, Z, F, G)

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        bX = gradients[0]
        if isinstance(bX.type, theano.gradient.DisconnectedType):
            bX = tt.zeros_like(outputs[0])
        return self.rev_op(*chain(inputs, outputs, [bX]))


class SolveRevOp(BaseOp):
    idim = (2, 2, 1, 2, "rhs", "rhs", "rhs", 2, 2, "rhs")
    odim = (2, 2, 1, 2, "rhs")
    dim_map = {"rhs": (1, 2)}

    def infer_shape(self, node, shapes):
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y, X, Z, F, G, bX = inputs
        N, J = U.shape

        bU = _resize_or_set(outputs, 0, (N, J))
        bP = _resize_or_set(outputs, 1, (N - 1, J))
        bd = _resize_or_set(outputs, 2, (N,))
        bW = _resize_or_set(outputs, 3, (N, J))

        if Y.ndim == 1:
            bY = _resize_or_set(outputs, 4, (N,))
        else:
            nrhs = Y.shape[1]
            bY = _resize_or_set(outputs, 4, (N, nrhs))

        backprop.solve_rev(U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY)


solve = SolveOp()


class NormOp(BaseOp):
    idim = (2, 2, 1, 2, 1)
    odim = (0, 1, 2)

    def __init__(self):
        self.rev_op = NormRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        return [], shapes[4], shapes[0]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y = inputs
        N, J = U.shape

        X = _resize_or_set(outputs, 0, ())
        Z = _resize_or_set(outputs, 1, (N,))
        F = _resize_or_set(outputs, 2, (N, J))

        backprop.norm_fwd(U, P, d, W, Y, X, Z, F)

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        bX = gradients[0]
        if isinstance(bX.type, theano.gradient.DisconnectedType):
            bX = tt.zeros_like(outputs[0])
        return self.rev_op(*chain(inputs, outputs, [bX]))


class NormRevOp(BaseOp):
    idim = (2, 2, 1, 2, 1, 0, 1, 2, 0)
    odim = (2, 2, 1, 2, 1)

    def infer_shape(self, node, shapes):
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y, X, Z, F, bX = inputs
        N, J = U.shape

        bU = _resize_or_set(outputs, 0, (N, J))
        bP = _resize_or_set(outputs, 1, (N - 1, J))
        bd = _resize_or_set(outputs, 2, (N,))
        bW = _resize_or_set(outputs, 3, (N, J))
        bY = _resize_or_set(outputs, 4, (N,))

        backprop.norm_rev(U, P, d, W, Y, X, Z, F, bX, bU, bP, bd, bW, bY)


norm = NormOp()


class DotTrilOp(BaseOp):
    idim = (2, 2, 1, 2, "rhs")
    odim = ("rhs", 2)
    dim_map = {"rhs": (1, 2)}

    def __init__(self):
        self.rev_op = DotTrilRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        N = shapes[0][0]
        J = shapes[0][1]
        try:
            nrhs = shapes[4][1]
        except IndexError:
            nrhs = 1
        return shapes[4], [N, nrhs * J]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y = inputs
        N, J = U.shape

        if Y.ndim == 1:
            X = _resize_or_set(outputs, 0, (N,))
            F = _resize_or_set(outputs, 1, (N, J))
        else:
            nrhs = Y.shape[1]
            X = _resize_or_set(outputs, 0, (N, nrhs))
            F = _resize_or_set(outputs, 1, (N, nrhs * J))

        backprop.dot_tril_fwd(U, P, d, W, Y, X, F)

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        bX = gradients[0]
        if isinstance(bX.type, theano.gradient.DisconnectedType):
            bX = tt.zeros_like(outputs[0])
        return self.rev_op(*chain(inputs, outputs, [bX]))


class DotTrilRevOp(BaseOp):
    idim = (2, 2, 1, 2, "rhs", "rhs", 2, "rhs")
    odim = (2, 2, 1, 2, "rhs")
    dim_map = {"rhs": (1, 2)}

    def infer_shape(self, node, shapes):
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y, X, F, bX = inputs
        N, J = U.shape

        bU = _resize_or_set(outputs, 0, (N, J))
        bP = _resize_or_set(outputs, 1, (N - 1, J))
        bd = _resize_or_set(outputs, 2, (N,))
        bW = _resize_or_set(outputs, 3, (N, J))

        if Y.ndim == 1:
            bY = _resize_or_set(outputs, 4, (N,))
        else:
            nrhs = Y.shape[1]
            bY = _resize_or_set(outputs, 4, (N, nrhs))

        backprop.dot_tril_rev(U, P, d, W, Y, X, F, bX, bU, bP, bd, bW, bY)


dot_tril = DotTrilOp()


class MatmulOp(BaseOp):
    idim = (1, 2, 2, 2, "rhs")
    odim = ("rhs", "rhs", 2, 2)
    dim_map = {"rhs": (1, 2)}

    def __init__(self):
        self.rev_op = MatmulRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        N = shapes[1][0]
        J = shapes[1][1]
        try:
            nrhs = shapes[4][1]
        except IndexError:
            nrhs = 1
        return shapes[4], shapes[4], [N, nrhs * J], [N, nrhs * J]

    def perform(self, node, inputs, outputs):
        a, U, V, P, Y = inputs
        N, J = U.shape
        if Y.ndim == 1:
            X = _resize_or_set(outputs, 0, (N,))
            Z = _resize_or_set(outputs, 1, (N,))
            F = _resize_or_set(outputs, 2, (N, J))
            G = _resize_or_set(outputs, 3, (N, J))
        else:
            nrhs = Y.shape[1]
            X = _resize_or_set(outputs, 0, (N, nrhs))
            Z = _resize_or_set(outputs, 1, (N, nrhs))
            F = _resize_or_set(outputs, 2, (N, nrhs * J))
            G = _resize_or_set(outputs, 3, (N, nrhs * J))

        backprop.matmul_fwd(a, U, V, P, Y, X, Z, F, G)

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        bX = gradients[0]
        if isinstance(bX.type, theano.gradient.DisconnectedType):
            bX = tt.zeros_like(outputs[0])
        return self.rev_op(*chain(inputs, outputs, [bX]))


class MatmulRevOp(BaseOp):
    idim = (1, 2, 2, 2, "rhs", "rhs", "rhs", 2, 2, "rhs")
    odim = (1, 2, 2, 2, "rhs")
    dim_map = {"rhs": (1, 2)}

    def infer_shape(self, node, shapes):
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        a, U, V, P, Y, X, Z, F, G, bX = inputs
        N, J = U.shape

        ba = _resize_or_set(outputs, 0, (N,))
        bU = _resize_or_set(outputs, 1, (N, J))
        bV = _resize_or_set(outputs, 2, (N, J))
        bP = _resize_or_set(outputs, 3, (N - 1, J))

        if Y.ndim == 1:
            bY = _resize_or_set(outputs, 4, (N,))
        else:
            nrhs = Y.shape[1]
            bY = _resize_or_set(outputs, 4, (N, nrhs))

        backprop.matmul_rev(a, U, V, P, Y, X, Z, F, G, bX, ba, bU, bV, bP, bY)


matmul = MatmulOp()


class ConditionalMeanOp(BaseOp):
    idim = (2, 2, 2, 1, 2, 2, 1)
    odim = (1,)
    dtypes = (
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int64",
    )

    def infer_shape(self, node, shapes):
        return (shapes[6],)

    def perform(self, node, inputs, outputs):
        U, V, P, z, U_star, V_star, inds = inputs
        mu = _resize_or_set(outputs, 0, inds.shape)
        driver.conditional_mean(U, V, P, z, U_star, V_star, inds, mu)


conditional_mean = ConditionalMeanOp()
