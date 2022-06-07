# -*- coding: utf-8 -*-

__all__ = ["get_matrices"]

import numpy as np

import celerite2.terms as terms


def get_matrices(
    size=100,
    kernel=None,
    vector=False,
    conditional=False,
    include_dense=False,
    no_diag=False,
):
    random = np.random.default_rng(721)
    x = np.sort(random.uniform(0, 10, size))
    if vector:
        Y = np.sin(x)
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x**2]).T, dtype=np.float64
        )
    if no_diag:
        diag = np.zeros_like(x)
    else:
        diag = random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    c, a, U, V = kernel.get_celerite_matrices(x, diag)

    if include_dense:
        K = kernel.get_value(x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += diag

    if not conditional:
        if include_dense:
            return x, c, a, U, V, K, Y
        return x, c, a, U, V, Y

    t = np.sort(random.uniform(-1, 12, 200))
    _, _, U2, V2 = kernel.get_celerite_matrices(t, np.zeros_like(t))

    if include_dense:
        K_star = kernel.get_value(t[:, None] - x[None, :])
        return x, c, a, U, V, K, Y, t, U2, V2, K_star

    return x, c, a, U, V, Y, t, U2, V2


def allclose(a, b, **kwargs):
    return a.shape == b.shape and np.allclose(a, b, **kwargs)


def _compare_tensor(eval_func, tensor, array, text, atol=1e-8):
    value = eval_func(tensor)
    if array.size == 0:
        assert value.size == 0
        return
    resid = np.abs(array - value)
    coords = np.unravel_index(np.argmax(resid), array.shape)
    np.testing.assert_allclose(
        value,
        array,
        atol=atol,
        err_msg=f"resid: {resid.max()}; coords: {coords}; message: {text}",
    )


def check_tensor_term(eval_func, term, pyterm, atol=1e-8):
    try:
        tensors = term.get_coefficients()
    except NotImplementedError:
        # Some terms don't implement coefficients
        pass
    else:
        arrays = pyterm.get_coefficients()
        inds = np.argsort(eval_func(tensors[0]))
        pyinds = np.argsort(arrays[0])
        for n, (tensor, array) in enumerate(zip(tensors[:2], arrays[:2])):
            _compare_tensor(
                eval_func,
                tensor[inds],
                array[pyinds],
                f"real coefficients {n}",
                atol=atol,
            )

        inds = np.argsort(eval_func(tensors[2]))
        pyinds = np.argsort(arrays[2])
        for n, (tensor, array) in enumerate(zip(tensors[2:], arrays[2:])):
            _compare_tensor(
                eval_func,
                tensor[inds],
                array[pyinds],
                f"complex coefficients {n}",
                atol=atol,
            )

    random = np.random.default_rng(40582)
    x = np.sort(random.uniform(0, 10, 50))
    diag = random.uniform(0.1, 0.3, len(x))

    # FIXME: Do we want to compare the actual celerite matrices that are
    # produced? This has always been horrible, and probably doesn't really
    # matter.

    # # This is a hack to deal with the fact that the interfaces don't
    # # always propduce matrices with the same column order
    # tensors = term.get_celerite_matrices(x, diag)
    # arrays = pyterm.get_celerite_matrices(x, diag)
    # inds = np.argsort(eval_func(tensors[2][1]))
    # pyinds = np.argsort(arrays[2][1])
    # for n, (tensor, array) in enumerate(zip(tensors, arrays)):
    #     if n == 0:
    #         _compare_tensor(
    #             eval_func,
    #             tensor[inds],
    #             array[pyinds],
    #             f"matrix {n}",
    #             atol=atol,
    #         )
    #     elif n == 1:
    #         _compare_tensor(
    #             eval_func,
    #             tensor,
    #             array,
    #             f"matrix {n}",
    #             atol=atol,
    #         )
    #     else:
    #         _compare_tensor(
    #             eval_func,
    #             tensor[:, inds],
    #             array[:, pyinds],
    #             f"matrix {n}",
    #             atol=atol,
    #         )

    _compare_tensor(
        eval_func,
        term.to_dense(x, diag),
        pyterm.to_dense(x, diag),
        "to_dense",
        atol=atol,
    )

    tau = x[:, None] - x[None, :]
    _compare_tensor(
        eval_func,
        term.get_value(tau),
        pyterm.get_value(tau),
        "get_value",
        atol=atol,
    )

    omega = np.linspace(-10, 10, 500)
    try:
        psd = term.get_psd(omega)
    except NotImplementedError:
        # Some terms don't implement the `get_psd` function
        pass
    else:
        _compare_tensor(
            eval_func,
            psd,
            pyterm.get_psd(omega),
            "get_psd",
            atol=atol,
        )

    y = np.reshape(np.sin(x), (len(x), 1))
    _compare_tensor(
        eval_func,
        term.dot(x, diag, y),
        pyterm.dot(x, diag, y),
        "dot",
        atol=atol,
    )


def check_gp_models(eval_func, gp, pygp, y, t):
    # "log_likelihood" method
    assert allclose(pygp.log_likelihood(y), eval_func(gp.log_likelihood(y)))

    # "condition" method
    pycond = pygp.condition(y)
    cond = gp.condition(y)
    assert allclose(pycond.mean, eval_func(cond.mean))
    assert allclose(pycond.variance, eval_func(cond.variance))
    assert allclose(pycond.covariance, eval_func(cond.covariance))

    pycond = pygp.condition(y, t=t)
    cond = gp.condition(y, t=t)
    assert allclose(pycond.mean, eval_func(cond.mean))
    assert allclose(pycond.variance, eval_func(cond.variance))
    assert allclose(pycond.covariance, eval_func(cond.covariance))

    # "dot_tril" method
    assert allclose(pygp.dot_tril(y), eval_func(gp.dot_tril(y)))
