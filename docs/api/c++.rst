.. _cpp-api:

C++
===

Basic interface
---------------

.. doxygenfunction:: celerite2::core::matmul(const Eigen::MatrixBase<Diag> &a, const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &V, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &X_out)

Autodiff Interface
------------------

.. doxygenfunction:: celerite2::core::matmul(const Eigen::MatrixBase<Diag> &a, const Eigen::MatrixBase<LowRank> &U, const Eigen::MatrixBase<LowRank> &V, const Eigen::MatrixBase<LowRank> &P, const Eigen::MatrixBase<RightHandSide> &Y, Eigen::MatrixBase<RightHandSideOut> const &X_out, Eigen::MatrixBase<RightHandSideOut> const &M_out, Eigen::MatrixBase<Work> const &F_out, Eigen::MatrixBase<Work> const &G_out)

Terms
-----

.. doxygenclass:: celerite2::Term
