.. _cpp-api:

C++ interface
=============

All of the computationally expensive parts of *celerite2* are implemented in
C++. This interface was not designed to be as user friendly as the Python
interfaces and it isn't as thoroughly documented, but these details should be
enough to get you started if you're interested in using or contributing to the
low-level code directly. Please feel free to `open an issue
<https://github.com/exoplanet-dev/celerite2/issues>`_ if you have questions!

One thing to note when using these functions is that the performance will be
greatly improved (especially for small values of ``J``) by using fixed size
matrices because `Eigen <http://eigen.tuxfamily.org>`_ can work its magic. If
you use the :ref:`cpp-terms` interface defined below to construct your low rank
matrices, the results will be right (using the `Width` parameter), but you might
need to enforce this manually if you're building your own matrices. Much of the
code in the `pybind11 <https://pybind11.readthedocs.io>`_ interface is dedicated
to making sure that small matrices correctly typed.

It is also worth noting that the notation in the equations below (and throughout
these pages) is a little sloppy since it generally neglects the ``P`` matrix.
Take a look at `the original celerite paper <https://arxiv.org/abs/1703.09710>`_
for more details about the numerical stability introduced by using this matrix,
and how it affects the algorithms.

.. _cpp-basic:

Basic interface
---------------

If you don't need access to the backpropagation functions, you can use the
following functions to compute the standard celerite operations. If you need
support for backpropagation, use the functions defined in the
:ref:`cpp-autodiff` section, but know that they will require a larger memory
footprint.

.. doxygenfunction:: celerite2::core::to_dense
.. doxygenfunction:: celerite2::core::factor(const Eigen::MatrixBase<Diag> &a, const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &V, const Eigen::MatrixBase<LowRank> &P, Eigen::MatrixBase<DiagOut> const &d_out, Eigen::MatrixBase<LowRankOut> const &W_out)
.. doxygenfunction:: celerite2::core::solve(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W,  const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &X_out)
.. doxygenfunction:: celerite2::core::norm(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<Norm> const &norm_out, Eigen::MatrixBase<RightHandSideOut> const &X_out)
.. doxygenfunction:: celerite2::core::dot_tril(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &Z_out)
.. doxygenfunction:: celerite2::core::conditional_mean


.. _cpp-autodiff:

Autodiff Interface
------------------

If you don't need access to the backpropagation functions, you should use the
functions defined in the :ref:`cpp-basic` section above. The functions defined
in this section track the extra matrices that are required for the reverse pass
in reverse-mode automatic differentiation. See `Foreman-Mackey (2018)
<https://arxiv.org/abs/1801.10156>`_ for more information about the reverse pass
calculations.

Each of these functions follows the same argument naming conventions. As an
example, imagine that we have a function called ``func`` that has two input
arguments, ``x`` (lowercase variables are vectors) and ``Y`` (and uppercase
variables are matrices), and and two outputs ``A`` and ``b``. Let's say that the
reverse pass also requires a cached variable (or "workspace") called ``S``. In
this case, the signature of the forward pass would be something like:

.. code-block:: cpp

    template <typename VectorIn, typename MatrixIn, typename MatrixOut,
              typename VectorOut, typename Workspace>
    void func(
      const Eigen::MatrixBase<VectorIn> &x,
      const Eigen::MatrixBase<MatrixIn> &Y,
      Eigen::MatrixBase<MatrixOut> const &A_out,
      Eigen::MatrixBase<VectorOut> const &b_out,
      Eigen::MatrixBase<Workspace> const &S_out
    )

Note that, in general, the ``_out`` parameters don't need to have the right
shape because they will be resized in place. The reverse pass of this function,
would be implemented in another function ``func_rev`` with the following
signature:

.. code-block:: cpp

    template <typename VectorIn, typename MatrixIn, typename Workspace,
              typename VectorOut, typename MatrixOut>
    void func_rev(
      // Original inputs
      const Eigen::MatrixBase<VectorIn> &x,
      const Eigen::MatrixBase<MatrixIn> &Y,
      // Original outputs
      const Eigen::MatrixBase<MatrixIn> &A,
      const Eigen::MatrixBase<VectorIn> &b,
      const Eigen::MatrixBase<Workspace> &S,
      // The sensitivities of the outputs, note: S is not included
      const Eigen::MatrixBase<MatrixIn> &bA,
      const Eigen::MatrixBase<VectorIn> &bb,
      // The (resulting) sensitivities of the inputs
      Eigen::MatrixBase<VectorOut> const &bx_out,
      Eigen::MatrixBase<MatrixOut> const &bY_out
    )

where the ``b`` prefix before a parameter indicates the overbar from the
notation in `Foreman-Mackey (2018) <https://arxiv.org/abs/1801.10156>`_:

.. math::

    \bar{x} = \frac{\partial \mathcal{L}}{\partial x}

The forward methods for each celerite operation are documented below, and the
reverse passes are all implemented following the rules listed above. Take a look
at `the source code <https://github.com/exoplanet-dev/celerite2>`_ to see the signatures.

.. doxygenfunction:: celerite2::core::factor(const Eigen::MatrixBase<Diag> &a, const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &V, const Eigen::MatrixBase<LowRank> &P, Eigen::MatrixBase<DiagOut> const &d_out, Eigen::MatrixBase<LowRankOut> const &W_out, Eigen::MatrixBase<Work> const &S_out)
.. doxygenfunction:: celerite2::core::solve(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &X_out, Eigen::MatrixBase<RightHandSideOut> const &Z_out, Eigen::MatrixBase<Work> const &F_out, Eigen::MatrixBase<Work> const &G_out)
.. doxygenfunction:: celerite2::core::norm(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<Norm> const &X_out, Eigen::MatrixBase<RightHandSideOut> const &Z_out, Eigen::MatrixBase<Work> const &F_out)
.. doxygenfunction:: celerite2::core::dot_tril(const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<Diag> &d, const Eigen::MatrixBase<LowRank> &W, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &Z_out, Eigen::MatrixBase<Work> const &F_out)
.. doxygenfunction:: celerite2::core::matmul(const Eigen::MatrixBase<Diag> &a, const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &V, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &X_out, Eigen::MatrixBase<RightHandSideOut> const &M_out, Eigen::MatrixBase<Work> const &F_out, Eigen::MatrixBase<Work> const &G_out)

.. _cpp-terms:

Terms
-----

The only the most basic terms are implemented in C++ and they are mostly used
for testing purposes, but it would be possible to generalize them to other use
cases.

.. doxygenclass:: celerite2::Term
.. doxygenclass:: celerite2::SHOTerm
.. doxygenclass:: celerite2::RealTerm
.. doxygenclass:: celerite2::ComplexTerm
