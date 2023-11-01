from pytensor.link.jax.dispatch import jax_funcify

from celerite2.jax import ops as jax_ops
from celerite2.pymc import ops as pymc_ops


@jax_funcify.register(pymc_ops._CeleriteOp)
def jax_funcify_Celerite(op, **kwargs):
    name = op.name
    if name.endswith("_fwd"):
        name = name[:-4] + "_p"
    else:
        assert name.endswith("_rev")
        name = name + "_p"
    prim = getattr(jax_ops, name)

    def impl(*args):
        return prim.bind(*args)

    return impl
