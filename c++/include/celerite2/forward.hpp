#ifndef _CELERITE2_FORWARD_HPP_DEFINED_
#define _CELERITE2_FORWARD_HPP_DEFINED_

#include <Eigen/Core>
#include "internal.hpp"
namespace celerite2 {
namespace core {

/**
 * \brief Get the dense representation of a celerite matrix
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param K_out (N, N): The dense matrix
 */
template <typename Input, typename Coeffs, typename Diag, typename LowRank, typename Dense>
void to_dense(const Eigen::MatrixBase<Input> &t,    // (N,)
              const Eigen::MatrixBase<Coeffs> &c,   // (J,)
              const Eigen::MatrixBase<Diag> &a,     // (N,)
              const Eigen::MatrixBase<LowRank> &U,  // (N, J)
              const Eigen::MatrixBase<LowRank> &V,  // (N, J)
              Eigen::MatrixBase<Dense> const &K_out // (N,)
) {
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;

  Eigen::Index N = U.rows(), J = U.cols();
  CAST_MAT(Dense, K, N, N);

  RowVector p(1, J);
  for (Eigen::Index m = 0; m < N; ++m) {
    p.setConstant(1.0);
    K(m, m) = a(m);
    for (Eigen::Index n = m + 1; n < N; ++n) {
      p.array() *= exp(-c.array() * (t(n) - t(n - 1)));
      K(n, m) = (U.row(n).array() * V.row(m).array() * p.array()).sum();
      K(m, n) = K(n, m);
    }
  }
}

/**
 * \brief Compute the Cholesky factorization of the system
 *
 * This computes `d` and `W` such that:
 *
 * `K = L*diag(d)*L^T`
 *
 * where `K` is the celerite matrix and
 *
 * `L = 1 + tril(U*W^T)`
 *
 * This can be safely applied in place: `d_out` can point to `a` and `W_out` can
 * point to `V`, and the memory will be reused. In this particular case, the
 * `celerite2::core::factor_rev` function doesn't use `a` and `V`, but this
 * won't be true for all `_rev` functions.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param d_out (N,): The diagonal component of the Cholesky factor
 * @param W_out (N, J): The second low rank component of the Cholesky factor
 * @param S_out (N, J*J): The cached value of the S matrix at each step
 */
template <bool update_workspace = true, typename Input, typename Coeffs, typename Diag, typename LowRank, typename DiagOut, typename LowRankOut,
          typename Work>
Eigen::Index factor(const Eigen::MatrixBase<Input> &t,          // (N,)
                    const Eigen::MatrixBase<Coeffs> &c,         // (J,)
                    const Eigen::MatrixBase<Diag> &a,           // (N,)
                    const Eigen::MatrixBase<LowRank> &U,        // (N, J)
                    const Eigen::MatrixBase<LowRank> &V,        // (N, J)
                    Eigen::MatrixBase<DiagOut> const &d_out,    // (N,)
                    Eigen::MatrixBase<LowRankOut> const &W_out, // (N, J)
                    Eigen::MatrixBase<Work> const &S_out        // (N, J*J)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename Diag::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;
  typedef typename Eigen::internal::plain_col_type<Coeffs>::type CoeffVector;

  Eigen::Index N = U.rows(), J = U.cols();
  CAST_VEC(DiagOut, d, N);
  CAST_MAT(LowRankOut, W, N, J);
  CAST_BASE(Work, S);
  if (update_workspace) {
    S.derived().resize(N, J * J);
    S.row(0).setZero();
  }

  // This is a temporary vector used to minimize computations internally
  RowVector tmp;
  CoeffVector p;

  // This holds the accumulated value of the S matrix at each step
  Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, LowRank::ColsAtCompileTime, Eigen::ColMajor> Sn(J, J);

  // This is a flattened pointer to Sn that is used for copying the data
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Sn.data(), 1, J * J);

  // First row
  Sn.setZero();
  d(0)               = a(0);
  W.row(0).noalias() = V.row(0) / d(0);

  // The rest of the rows
  for (Eigen::Index n = 1; n < N; ++n) {
    p = exp(c.array() * (t(n - 1) - t(n)));

    // Update S_n = diag(P) * (S_n-1 + d*W*W.T) * diag(P)
    Sn.noalias() += d(n - 1) * W.row(n - 1).transpose() * W.row(n - 1);
    Sn = p.asDiagonal() * Sn;

    // Save the current value of Sn to the workspace
    // Note: This is actually `diag(P) * (S + d*W*W.T)` without the final `* diag(P)`
    internal::update_workspace<update_workspace>::apply(n, ptr, S);

    // Incorporate the second diag(P) that we didn't include above for bookkeeping
    Sn *= p.asDiagonal();

    // Update d = a - U * S * U.T
    tmp  = U.row(n) * Sn;
    d(n) = a(n) - tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() = (V.row(n) - tmp) / d(n);
  }

  return 0;
}

/**
 * \brief Apply a strictly lower matrix multiply
 *
 * This computes:
 *
 * `Z += tril(U * V^T) * Y`
 *
 * where `tril` is the strictly lower triangular function.
 *
 * Note that this will *update* the value of `Z`.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param U     (N, J): The first low rank matrix
 * @param W     (N, J): The second low rank matrix
 * @param Y     (N, Nrhs): The matrix to be multiplied
 * @param Z_out (N, Nrhs): The matrix to be updated
 * @param F_out (N, J*Nrhs): The workspace
 */
template <bool update_workspace = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void solve_lower(const Eigen::MatrixBase<Input> &t,                // (N,)
                 const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                 const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                 const Eigen::MatrixBase<LowRank> &W,              // (N, J)
                 const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
                 Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                 Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);
  CAST_BASE(RightHandSideOut, Z);
  Z = Y;
  internal::forward<true, update_workspace>(t, c, U, W, Y, Z, F_out);
}

/**
 * \brief Compute the solution of a upper triangular linear equation
 *
 * This computes `Z` such that:
 *
 * `Y = L^T * Y`
 *
 * where
 *
 * `L = 1 + tril(U*W^T)`
 *
 * This can be safely applied in place.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param U     (N, J): The first low rank matrix
 * @param W     (N, J): The second low rank matrix
 * @param Y     (N, Nrhs): The right hand side
 * @param Z_out (N, Nrhs): The solution of this equation
 * @param F_out (N, J*Nrhs): The workspace
 */
template <bool update_workspace = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void solve_upper(const Eigen::MatrixBase<Input> &t,                // (N,)
                 const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                 const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                 const Eigen::MatrixBase<LowRank> &W,              // (N, J)
                 const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
                 Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                 Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);
  CAST_BASE(RightHandSideOut, Z);
  Z = Y;
  internal::backward<true, update_workspace>(t, c, U, W, Y, Z, F_out);
}

/**
 * \brief Apply a strictly lower matrix multiply
 *
 * This computes:
 *
 * `Z += tril(U * V^T) * Y`
 *
 * where `tril` is the strictly lower triangular function.
 *
 * Note that this will *update* the value of `Z`.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param Y     (N, Nrhs): The matrix to be multiplied
 * @param Z_out (N, Nrhs): The matrix to be updated
 * @param F_out (N, J*Nrhs): The workspace
 */
template <bool update_workspace = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void matmul_lower(const Eigen::MatrixBase<Input> &t,                // (N,)
                  const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                  const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                  const Eigen::MatrixBase<LowRank> &V,              // (N, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
                  Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                  Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  internal::forward<false, update_workspace>(t, c, U, V, Y, Z_out, F_out);
}

/**
 * \brief Apply a strictly upper matrix multiply
 *
 * This computes:
 *
 * `Z += triu(V * U^T) * Y`
 *
 * where `triu` is the strictly lower triangular function.
 *
 * Note that this will *update* the value of `Z`.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param Y     (N, Nrhs): The matrix to be multiplied
 * @param Z_out (N, Nrhs): The matrix to be updated
 * @param F_out (N, J*Nrhs): The workspace
 */
template <bool update_workspace = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void matmul_upper(const Eigen::MatrixBase<Input> &t,                // (N,)
                  const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                  const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                  const Eigen::MatrixBase<LowRank> &V,              // (N, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
                  Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                  Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  internal::backward<false, update_workspace>(t, c, U, V, Y, Z_out, F_out);
}

/**
 * \brief The general lower-triangular dot product of a rectangular celerite system
 *
 * @param t1     (N,): The left input coordinates (must be sorted)
 * @param t2     (M,): The right input coordinates (must be sorted)
 * @param c      (J,): The transport coefficients
 * @param U      (N, J): The first low rank matrix
 * @param V      (M, J): The second low rank matrix
 * @param Y      (M, Nrhs): The matrix that will be left multiplied by the celerite model
 * @param Z_out  (N, Nrhs): The result of the operation
 * @param F_out  (M, J*Nrhs): The workspace
 */
template <bool do_update = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void general_matmul_lower(const Eigen::MatrixBase<Input> &t1,               // (N,)
                          const Eigen::MatrixBase<Input> &t2,               // (M,)
                          const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                          const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                          const Eigen::MatrixBase<LowRank> &V,              // (M, J)
                          const Eigen::MatrixBase<RightHandSide> &Y,        // (M, nrhs)
                          Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                          Eigen::MatrixBase<Work> const &F_out              // (M, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::internal::plain_col_type<Coeffs>::type CoeffVector;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = t1.rows(), M = t2.rows(), J = c.rows(), nrhs = Y.cols();

  CAST_BASE(RightHandSideOut, Z);
  CAST_BASE(Work, F);
  if (do_update) {
    F.derived().resize(M, J * nrhs);
    F.row(0).setZero();
  }

  CoeffVector p(J);
  Inner Fm = V.row(0).transpose() * Y.row(0);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fm.data(), 1, J * nrhs);
  internal::update_workspace<do_update>::apply(0, ptr, F);

  Scalar tn = t2(0);
  Eigen::Index n, m = 1;
  for (n = 0; n < N; ++n)
    if (t1(n) >= tn) break;
  for (; n < N; ++n) {
    tn = t1(n);
    while (m < M && t2(m) <= tn) {
      p  = exp(c.array() * (t2(m - 1) - t2(m)));
      Fm = p.asDiagonal() * Fm;
      Fm.noalias() += V.row(m).transpose() * Y.row(m);
      internal::update_workspace<do_update>::apply(m, ptr, F);
      m++;
    }
    p = exp(c.array() * (t2(m - 1) - tn));
    Z.row(n).noalias() += U.row(n) * p.asDiagonal() * Fm;
  }
}

/**
 * \brief The general upper-triangular dot product of a rectangular celerite system
 *
 * @param t1     (N,): The left input coordinates (must be sorted)
 * @param t2     (M,): The right input coordinates (must be sorted)
 * @param c      (J,): The transport coefficients
 * @param U      (N, J): The first low rank matrix
 * @param V      (M, J): The second low rank matrix
 * @param Y      (M, Nrhs): The matrix that will be left multiplied by the celerite model
 * @param Z_out  (N, Nrhs): The result of the operation
 * @param F_out  (M, J*Nrhs): The workspace
 */
template <bool do_update = true, typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut,
          typename Work>
void general_matmul_upper(const Eigen::MatrixBase<Input> &t1,               // (N,)
                          const Eigen::MatrixBase<Input> &t2,               // (M,)
                          const Eigen::MatrixBase<Coeffs> &c,               // (J,)
                          const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                          const Eigen::MatrixBase<LowRank> &V,              // (M, J)
                          const Eigen::MatrixBase<RightHandSide> &Y,        // (M, nrhs)
                          Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
                          Eigen::MatrixBase<Work> const &F_out              // (M, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::internal::plain_col_type<Coeffs>::type CoeffVector;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = t1.rows(), M = t2.rows(), J = c.rows(), nrhs = Y.cols();

  CAST_BASE(RightHandSideOut, Z);
  CAST_BASE(Work, F);
  if (do_update) {
    F.derived().resize(M, J * nrhs);
    F.row(0).setZero();
  }

  CoeffVector p(J);
  Inner Fm = V.row(M - 1).transpose() * Y.row(M - 1);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fm.data(), 1, J * nrhs);

  Scalar tn = t2(M - 1);
  Eigen::Index n, m = M - 2;
  for (n = N - 1; n >= 0; --n)
    if (t1(n) < tn) break;
  for (; n >= 0; --n) {
    tn = t1(n);
    while (m >= 0 && t2(m) > tn) {
      p  = exp(c.array() * (t2(m) - t2(m + 1)));
      Fm = p.asDiagonal() * Fm;
      Fm.noalias() += V.row(m).transpose() * Y.row(m);
      internal::update_workspace<do_update>::apply(m, ptr, F);
      m--;
    }
    p = exp(c.array() * (tn - t2(m + 1)));
    Z.row(n).noalias() += U.row(n) * p.asDiagonal() * Fm;
  }
}

} // namespace core
} // namespace celerite2

#endif // _CELERITE2_FORWARD_HPP_DEFINED_
