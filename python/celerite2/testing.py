# -*- coding: utf-8 -*-

__all__ = ["get_matrices"]
import numpy as np

from . import terms


def get_matrices(
    size=100, kernel=None, vector=False, conditional=False, include_dense=False
):
    np.random.seed(721)
    x = np.sort(np.random.uniform(0, 10, size))
    if vector:
        Y = np.sin(x)
    else:
        Y = np.ascontiguousarray(
            np.vstack([np.sin(x), np.cos(x), x ** 2]).T, dtype=np.float64
        )
    diag = np.random.uniform(0.1, 0.3, len(x))
    kernel = kernel if kernel else terms.SHOTerm(S0=5.0, w0=0.1, Q=3.45)
    a, U, V, P = kernel.get_celerite_matrices(x, diag)

    if include_dense:
        K = kernel.get_value(x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += diag

    if not conditional:
        if include_dense:
            return a, U, V, P, K, Y
        return a, U, V, P, Y

    t = np.sort(np.random.uniform(-1, 12, 200))
    U_star, V_star, inds = kernel.get_conditional_mean_matrices(x, t)

    if include_dense:
        K_star = kernel.get_value(t[:, None] - x[None, :])
        return a, U, V, P, K, Y, U_star, V_star, inds, K_star

    return a, U, V, P, Y, U_star, V_star, inds


def allclose(a, b, **kwargs):
    return a.shape == b.shape and np.allclose(a, b, **kwargs)


def _compare_tensor(eval_func, tensor, array, text, atol=1e-8):
    value = eval_func(tensor)
    if array.size == 0:
        assert value.size == 0
        return
    resid = np.abs(array - value)
    coords = np.unravel_index(np.argmax(resid), array.shape)
    assert allclose(
        value, array, atol=atol
    ), f"resid: {resid.max()}; coords: {coords}; message: {text}"


def check_tensor_term(eval_func, term, pyterm, atol=1e-8):
    tensors = term.get_coefficients()
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

    np.random.seed(40582)
    x = np.sort(np.random.uniform(0, 10, 50))
    t = np.random.uniform(-1, 11, 75)
    diag = np.random.uniform(0.1, 0.3, len(x))

    # This is a hack to deal with the fact that the torch interface doesn't
    # always propduce matrices with the same column order
    tensors = term.get_celerite_matrices(x, diag)
    arrays = pyterm.get_celerite_matrices(x, diag)
    inds = np.argsort(eval_func(tensors[1])[0, :])
    pyinds = np.argsort(arrays[1][0, :])
    for n, (tensor, array) in enumerate(zip(tensors, arrays)):
        if n >= 1:
            _compare_tensor(
                eval_func,
                tensor[:, inds],
                array[:, pyinds],
                f"matrix {n}",
                atol=atol,
            )
        else:
            _compare_tensor(
                eval_func, tensor, array, f"matrix {n}", atol=atol,
            )

    # Same hack again...
    tensors = term.get_conditional_mean_matrices(x, t)
    arrays = pyterm.get_conditional_mean_matrices(x, t)
    _compare_tensor(
        eval_func, tensors[-1], arrays[-1], "sorted inds", atol=atol,
    )
    for n, (tensor, array) in enumerate(zip(tensors[:2], arrays[:2])):
        _compare_tensor(
            eval_func,
            tensor[:, inds],
            array[:, pyinds],
            f"conditional matrix {n}",
            atol=atol,
        )

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
    _compare_tensor(
        eval_func,
        term.get_psd(omega),
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

    # "predict" method
    for flag, args in [
        (False, dict(return_cov=False, return_var=False)),
        (True, dict(return_cov=False, return_var=True)),
        (True, dict(return_cov=True, return_var=False)),
    ]:
        if flag:
            assert all(
                allclose(a, eval_func(b))
                for a, b in zip(
                    pygp.predict(y, **args), gp.predict(y, **args),
                )
            )
            assert all(
                allclose(a, eval_func(b))
                for a, b in zip(
                    pygp.predict(y, t=t, **args), gp.predict(y, t=t, **args),
                )
            )
        else:
            assert allclose(
                pygp.predict(y, **args), eval_func(gp.predict(y, **args))
            )
            assert allclose(
                pygp.predict(y, t=t, **args),
                eval_func(gp.predict(y, t=t, **args)),
            )

    # "dot_tril" method
    assert allclose(pygp.dot_tril(y), eval_func(gp.dot_tril(y)))
