#ifndef _CELERITE2_FORWARD_HPP_DEFINED_
#define _CELERITE2_FORWARD_HPP_DEFINED_

#include <Eigen/Core>
#include "internal.hpp"
namespace celerite2 {
namespace core {

/**
 * \brief Get the dense representation of a celerite matrix
 *
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param K_out (N, N): The dense matrix
 */
template <typename Diag, typename LowRank, typename Dense>
void to_dense(const Eigen::MatrixBase<Diag> &a,     // (N,)
              const Eigen::MatrixBase<LowRank> &U,  // (N, J)
              const Eigen::MatrixBase<LowRank> &V,  // (N, J)
              const Eigen::MatrixBase<LowRank> &P,  // (N-1, J)
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
      p.array() *= P.row(n - 1).array();
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
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param d_out (N,): The diagonal component of the Cholesky factor
 * @param W_out (N, J): The second low rank component of the Cholesky factor
 * @param S_out (N, J*J): The cached value of the S matrix at each step
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename DiagOut, typename LowRankOut, typename Work>
Eigen::Index factor(const Eigen::MatrixBase<Diag> &a,           // (N,)
                    const Eigen::MatrixBase<LowRank> &U,        // (N, J)
                    const Eigen::MatrixBase<LowRank> &V,        // (N, J)
                    const Eigen::MatrixBase<LowRank> &P,        // (N-1, J)
                    Eigen::MatrixBase<DiagOut> const &d_out,    // (N,)
                    Eigen::MatrixBase<LowRankOut> const &W_out, // (N, J)
                    Eigen::MatrixBase<Work> const &S_out        // (N, J*J)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename Diag::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;

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
    // Update S_n = diag(P) * (S_n-1 + d*W*W.T) * diag(P)
    Sn.noalias() += d(n - 1) * W.row(n - 1).transpose() * W.row(n - 1);
    Sn = P.row(n - 1).asDiagonal() * Sn;

    // Save the current value of Sn to the workspace
    // Note: This is actually `diag(P) * (S + d*W*W.T)` without the final `* diag(P)`
    internal::update_workspace<update_workspace>::apply(n, ptr, S);

    // Incorporate the second diag(P) that we didn't include above for bookkeeping
    Sn *= P.row(n - 1).asDiagonal();

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
 * \brief Solve a linear system using the Cholesky factorization
 *
 * This computes `X` in the following linear system:
 *
 * `K * X = Y`
 *
 * where `K` is the celerite matrix. This uses the results of the Cholesky
 * factorization implemented by `celerite2::core::factor`.
 *
 * This can be safely applied in place *as long as you don't need to compute the
 * reverse pass*. To compute the solve in place, set `X_out = Y` and `Z_out = Y`.
 *
 * @param U     (N, J): The first low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param d     (N,): The diagonal component of the Cholesky factor
 * @param W     (N, J): The second low rank component of the Cholesky factor
 * @param Y     (N, Nrhs): The right hand side vector or matrix
 * @param X_out (N, Nrhs): The final solution to the linear system
 * @param Z_out (N, Nrhs): An intermediate result of the operation
 * @param F_out (N, J*Nrhs): The workspace for the forward sweep
 * @param G_out (N, J*Nrhs): The workspace for the backward sweep
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut, typename Work>
void solve(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
           const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
           const Eigen::MatrixBase<Diag> &d,                 // (N,)
           const Eigen::MatrixBase<LowRank> &W,              // (N, J)
           const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
           Eigen::MatrixBase<RightHandSideOut> const &X_out, // (N, nrhs)
           Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
           Eigen::MatrixBase<Work> const &F_out,             // (N, J*nrhs)
           Eigen::MatrixBase<Work> const &G_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST_BASE(RightHandSideOut, X);
  CAST_BASE(RightHandSideOut, Z);

  Z = Y;
  internal::forward<true, update_workspace>(U, W, P, Y, Z, F_out);

  X = Z;
  X.array().colwise() /= d.array();
  internal::backward<true, update_workspace>(U, W, P, Z, X, G_out);
}

/**
 * \brief Compute the norm of vector or matrix under the celerite metric
 *
 * This computes `Y^T * K^-1 * Y` where `K` is the celerite matrix. This uses
 * the results of the Cholesky factorization implemented by
 * `celerite2::core::factor`.
 *
 * This can be safely applied in place *as long as you don't need to compute the
 * reverse pass*. To compute the solve in place, set `Z_out = Y`.
 *
 * @param U     (N, J): The first low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param d     (N,): The diagonal component of the Cholesky factor
 * @param W     (N, J): The second low rank component of the Cholesky factor
 * @param Y     (N, Nrhs): The target vector or matrix
 * @param X_out (Nrhs, Nrhs): The norm of `Y`
 * @param Z_out (N, Nrhs): An intermediate result of the operation
 * @param F_out (N, J*Nrhs): The workspace for the forward sweep
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename Norm, typename RightHandSideOut,
          typename Work>
void norm(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
          const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
          const Eigen::MatrixBase<Diag> &d,                 // (N,)
          const Eigen::MatrixBase<LowRank> &W,              // (N, J)
          const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
          Eigen::MatrixBase<Norm> const &X_out,             // (nrhs, nrhs)
          Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
          Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  Eigen::Index nrhs = Y.cols();
  CAST_MAT(Norm, X, nrhs, nrhs);
  CAST_BASE(RightHandSideOut, Z);

  Z = Y;
  internal::forward<true, update_workspace>(U, W, P, Y, Z, F_out);

  X = Z.transpose() * d.asDiagonal().inverse() * Z;
}

/**
 * \brief Compute product of the Cholesky factor with a vector or matrix
 *
 * This computes `L * Y` where `L` is the Cholesky factor of a celerite system
 * computed using `celerite2::core::factor`.
 *
 * This can be safely applied in place *as long as you don't need to compute the
 * reverse pass*. To compute the solve in place, set `Z_out = Y`.
 *
 * @param U     (N, J): The first low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param d     (N,): The diagonal component of the Cholesky factor
 * @param W     (N, J): The second low rank component of the Cholesky factor
 * @param Y     (N, Nrhs): The target vector or matrix
 * @param Z_out (N, Nrhs): The result of the operation
 * @param F_out (N, J*Nrhs): The workspace for the forward sweep
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut, typename Work>
void dot_tril(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
              const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
              const Eigen::MatrixBase<Diag> &d,                 // (N,)
              const Eigen::MatrixBase<LowRank> &W,              // (N, J)
              const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
              Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, nrhs)
              Eigen::MatrixBase<Work> const &F_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);
  CAST_BASE(RightHandSideOut, Z);
  Z = Y;
  Z.array().colwise() *= sqrt(d.array());
  internal::forward<false, update_workspace>(U, W, P, Z, Z, F_out);
}

/**
 * \brief Compute a matrix-vector or matrix-matrix product
 *
 * This computes `X = K * Y` where `K` is the celerite matrix.
 *
 * Note that this operation *cannot* be safely applied in place.
 *
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param P     (N-1, J): The exponential difference matrix
 * @param Y     (N, Nrhs): The target vector or matrix
 * @param X_out (N, Nrhs): The result of the operation
 * @param M_out (N, Nrhs): The intermediate state of the system
 * @param F_out (N, J*Nrhs): The workspace for the forward sweep
 * @param G_out (N, J*Nrhs): The workspace for the backward sweep
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut, typename Work>
void matmul(const Eigen::MatrixBase<Diag> &a,                 // (N,)
            const Eigen::MatrixBase<LowRank> &U,              // (N, J)
            const Eigen::MatrixBase<LowRank> &V,              // (N, J)
            const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
            const Eigen::MatrixBase<RightHandSide> &Y,        // (N, nrhs)
            Eigen::MatrixBase<RightHandSideOut> const &X_out, // (N, nrhs)
            Eigen::MatrixBase<RightHandSideOut> const &M_out, // (N, nrhs)
            Eigen::MatrixBase<Work> const &F_out,             // (N, J*nrhs)
            Eigen::MatrixBase<Work> const &G_out              // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST_BASE(RightHandSideOut, X);
  CAST_BASE(RightHandSideOut, M);

  // M = diag(a) * Y + tril(U V^T) * Y
  M = a.asDiagonal() * Y;
  internal::forward<false, update_workspace>(U, V, P, Y, M, F_out);

  // X = M + triu(V U^T) * Y
  X = M;
  internal::backward<false, update_workspace>(U, V, P, Y, X, G_out);
}

/**
 * \brief Compute the conditional mean of a celerite process
 *
 * This computes `K_star * K^-1 * y` where `K` is the celerite matrix and
 * `K_star` is the rectangular covariance between the training points and the
 * test points.
 *
 * See the original celerite paper (https://arxiv.org/abs/1703.09710) for the
 * definitions of the `_star` matrices.
 *
 * Note that this operation *cannot* be safely applied in place.
 *
 * @param U      (N, J): The first low rank matrix
 * @param V      (N, J): The second low rank matrix
 * @param P      (N-1, J): The exponential difference matrix
 * @param z      (N,): The solution to the linear system `x = K^-1 * y`
 * @param U_star (M, J): The first low rank matrix for `K_star`
 * @param V_star (M, J): The second low rank matrix for `K_star`
 * @param inds   (M,): The indices where the test points would be inserted into
 *                     the training data; this *must* be sorted
 * @param mu_out (M,): The conditional mean
 */
template <typename LowRank, typename RightHandSide, typename Indices, typename RightHandSideOut>
void conditional_mean(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
                      const Eigen::MatrixBase<LowRank> &V,              // (N, J)
                      const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
                      const Eigen::MatrixBase<RightHandSide> &z,        // (N)  ->  The result of a solve
                      const Eigen::MatrixBase<LowRank> &U_star,         // (M, J)
                      const Eigen::MatrixBase<LowRank> &V_star,         // (M, J)
                      const Eigen::MatrixBase<Indices> &inds,           // (M)  ->  Index where the mth data point should be
                                                                        // inserted (the output of search_sorted)
                      Eigen::MatrixBase<RightHandSideOut> const &mu_out // (M)
) {
  Eigen::Index N = U.rows(), J = U.cols(), M = U_star.rows();
  CAST_VEC(RightHandSideOut, mu, M);
  Eigen::Matrix<typename LowRank::Scalar, 1, LowRank::ColsAtCompileTime> q(1, J);

  // Forward pass
  Eigen::Index m = 0;
  q.setZero();
  while (m < M && inds(m) <= 0) {
    mu(m) = 0;
    ++m;
  }
  for (Eigen::Index n = 0; n < N - 1; ++n) {
    q += z(n) * V.row(n);
    q *= P.row(n).asDiagonal();
    while ((m < M) && (inds(m) <= n + 1)) {
      mu(m) = U_star.row(m) * q.transpose();
      ++m;
    }
  }
  q += z(N - 1) * V.row(N - 1);
  while (m < M) {
    mu(m) = U_star.row(m) * q.transpose();
    ++m;
  }

  // Backward pass
  m = M - 1;
  q.setZero();
  while ((m >= 0) && (inds(m) > N - 1)) { --m; }
  for (Eigen::Index n = N - 1; n > 0; --n) {
    q += z(n) * U.row(n);
    q *= P.row(n - 1).asDiagonal();
    while ((m >= 0) && (inds(m) > n - 1)) {
      mu(m) += V_star.row(m) * q.transpose();
      --m;
    }
  }
  q += z(0) * U.row(0);
  while (m >= 0) {
    mu(m) = V_star.row(m) * q.transpose();
    --m;
  }
}

} // namespace core
} // namespace celerite2

#endif // _CELERITE2_FORWARD_HPP_DEFINED_
