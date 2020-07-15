# -*- coding: utf-8 -*-

__all__ = ["factor", "solve"]
from itertools import chain

import numpy as np
import theano
from theano import tensor as tt

from .. import backprop


def _resize_or_set(outputs, n, shape):
    if outputs[n][0] is None:
        outputs[n][0] = np.empty(shape)
    else:
        outputs[n][0] = np.ascontiguousarray(
            np.resize(outputs[n][0], shape), dtype=np.float64
        )
    return outputs[n][0]


class FactorOp(theano.Op):
    __props__ = ()
    itypes = [
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]
    otypes = [
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]

    def __init__(self):
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

        backprop.factor_fwd(a, U, V, P, d, W, S)

    def grad(self, inputs, gradients):
        outputs = self(*inputs)
        grads = (
            tt.zeros_like(outputs[n])
            if isinstance(b.type, theano.gradient.DisconnectedType)
            else b
            for n, b in enumerate(gradients[:2])
        )
        return self.rev_op(*chain(inputs, outputs, grads))


class FactorRevOp(theano.Op):
    __props__ = ()
    itypes = [
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dvector,
        theano.tensor.dmatrix,
    ]
    otypes = [
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]

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


class SolveOp(theano.Op):
    __props__ = ()
    itypes = [
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]
    otypes = [
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]

    def __init__(self):
        self.rev_op = SolveRevOp()
        super().__init__()

    def infer_shape(self, node, shapes):
        N = shapes[0][0]
        J = shapes[0][1]
        nrhs = shapes[4][1]
        return shapes[4], shapes[4], [N, nrhs * J], [N, nrhs * J]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y = inputs
        N, J = U.shape
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


class SolveRevOp(theano.Op):
    __props__ = ()
    itypes = [
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]
    otypes = [
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
        theano.tensor.dvector,
        theano.tensor.dmatrix,
        theano.tensor.dmatrix,
    ]

    def infer_shape(self, node, shapes):
        return shapes[:5]

    def perform(self, node, inputs, outputs):
        U, P, d, W, Y, X, Z, F, G, bX = inputs
        N, J = U.shape
        nrhs = Y.shape[1]

        bU = _resize_or_set(outputs, 0, (N, J))
        bP = _resize_or_set(outputs, 1, (N - 1, J))
        bd = _resize_or_set(outputs, 2, (N,))
        bW = _resize_or_set(outputs, 3, (N, J))
        bY = _resize_or_set(outputs, 4, (N, nrhs))

        backprop.solve_rev(U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY)


solve = SolveOp()
