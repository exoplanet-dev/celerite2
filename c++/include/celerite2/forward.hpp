#ifndef _CELERITE2_FORWARD_HPP_DEFINED_
#define _CELERITE2_FORWARD_HPP_DEFINED_

#include <Eigen/Core>
#include "internal.hpp"
namespace celerite2 {
namespace core {

template <typename Diag, typename LowRank, typename Dense>
void to_dense(const Eigen::MatrixBase<Diag> &a,     // (N,)
              const Eigen::MatrixBase<LowRank> &U,  // (N, J)
              const Eigen::MatrixBase<LowRank> &V,  // (N, J)
              const Eigen::MatrixBase<LowRank> &P,  // (N-1, J)
              Eigen::MatrixBase<Dense> const &K_out // (N,)
) {
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;

  int N = U.rows(), J = U.cols();
  CAST(Dense, K, N, N);

  RowVector p(1, J);
  for (int m = 0; m < N; ++m) {
    p.setConstant(1.0);
    K(m, m) = a(m);
    for (int n = m + 1; n < N; ++n) {
      p.array() *= P.row(n - 1).array();
      K(n, m) = (U.row(n).array() * V.row(m).array() * p.array()).sum();
      K(m, n) = K(n, m);
    }
  }
}

template <bool update_workspace = true, typename Diag, typename LowRank, typename Work>
int factor(const Eigen::MatrixBase<Diag> &a,        // (N,)
           const Eigen::MatrixBase<LowRank> &U,     // (N, J)
           const Eigen::MatrixBase<LowRank> &V,     // (N, J)
           const Eigen::MatrixBase<LowRank> &P,     // (N-1, J)
           Eigen::MatrixBase<Diag> const &d_out,    // (N,)
           Eigen::MatrixBase<LowRank> const &W_out, // (N, J)
           Eigen::MatrixBase<Work> const &S_out     // (N, J*J)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename Diag::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;

  int N = U.rows(), J = U.cols();
  CAST(Diag, d, N);
  CAST(LowRank, W, N, J);
  CAST(Work, S);
  if (update_workspace) { S.derived().resize(N, J * J); }

  // This is a temporary vector used to minimize computations internally
  RowVector tmp;

  // This holds the accumulated value of the S matrix at each step
  Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, LowRank::ColsAtCompileTime, Eigen::ColMajor> Sn(J, J);

  // This is a flattened pointer to Sn that is used for copying the data
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Sn.data(), 1, J * J);

  // First row
  S.row(0).setZero();
  Sn.setZero();
  d(0)               = a(0);
  W.row(0).noalias() = V.row(0) / d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
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

template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename Work>
void solve(const Eigen::MatrixBase<LowRank> &U,           // (N, J)
           const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
           const Eigen::MatrixBase<Diag> &d,              // (N,)
           const Eigen::MatrixBase<LowRank> &W,           // (N, J)
           const Eigen::MatrixBase<RightHandSide> &Y,     // (N, nrhs)
           Eigen::MatrixBase<RightHandSide> const &X_out, // (N, nrhs)
           Eigen::MatrixBase<RightHandSide> const &Z_out, // (N, nrhs)
           Eigen::MatrixBase<Work> const &F_out,          // (N, J*nrhs)
           Eigen::MatrixBase<Work> const &G_out           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, X);
  CAST(RightHandSide, Z);

  Z = Y;
  internal::forward<true, update_workspace>(U, W, P, Y, Z, F_out);

  X = Z;
  X.array().colwise() /= d.array();
  internal::backward<true, update_workspace>(U, W, P, Z, X, G_out);
}

template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename Work>
void dot_tril(const Eigen::MatrixBase<LowRank> &U,           // (N, J)
              const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
              const Eigen::MatrixBase<Diag> &d,              // (N,)
              const Eigen::MatrixBase<LowRank> &W,           // (N, J)
              const Eigen::MatrixBase<RightHandSide> &Y,     // (N, nrhs)
              Eigen::MatrixBase<RightHandSide> const &Z_out, // (N, nrhs)
              Eigen::MatrixBase<Work> const &F_out           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);
  CAST(RightHandSide, Z);
  Z = Y;
  Z.array().colwise() *= sqrt(d.array());
  internal::forward<false, update_workspace>(U, W, P, Z, Z, F_out);
}

/**
 * Matrix multiply
 *
 * This computes `X = [diag(a) + tril(U*V^T) + triu(V*U^T)] * Y` with `O(N*J^2)` scaling
 * with the `P` matrix from Foreman-Mackey et al. for numerical stability. Note that this
 * operation *cannot* be applied in-place.
 *
 * @param a      (N,): The diagonal component
 * @param U      (N, J): The first low rank matrix
 * @param V      (N, J): The second low rank matrix
 * @param P      (N - 1, J): The exponential difference matrix
 * @param Y      (N, Nrhs): The matrix that will be left multiplied by the celerite model
 * @param X_out  (N, Nrhs): The result of the operation
 * @param M_out  (N, Nrhs): The intermediate state of the system
 * @param F_out  (N, J * Nrhs): The workspace for the forward pass
 * @param G_out  (N, J * Nrhs): The workspace for the backward pass
 *
 */
template <bool update_workspace = true, typename Diag, typename LowRank, typename RightHandSide, typename Work>
void matmul(const Eigen::MatrixBase<Diag> &a,              // (N,)
            const Eigen::MatrixBase<LowRank> &U,           // (N, J)
            const Eigen::MatrixBase<LowRank> &V,           // (N, J)
            const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
            const Eigen::MatrixBase<RightHandSide> &Y,     // (N, nrhs)
            Eigen::MatrixBase<RightHandSide> const &X_out, // (N, nrhs)
            Eigen::MatrixBase<RightHandSide> const &M_out, // (N, nrhs)
            Eigen::MatrixBase<Work> const &F_out,          // (N, J*nrhs)
            Eigen::MatrixBase<Work> const &G_out           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, X);
  CAST(RightHandSide, M);

  // M = diag(a) * Y + tril(U V^T) * Y
  M = a.asDiagonal() * Y;
  internal::forward<false, update_workspace>(U, V, P, Y, M, F_out);

  // X = M + triu(V U^T) * Y
  X = M;
  internal::backward<false, update_workspace>(U, V, P, Y, X, G_out);
}

template <typename LowRank, typename RightHandSide, typename Indices>
void conditional_mean(const Eigen::MatrixBase<LowRank> &U,           // (N, J)
                      const Eigen::MatrixBase<LowRank> &V,           // (N, J)
                      const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
                      const Eigen::MatrixBase<RightHandSide> &z,     // (N)  ->  The result of a solve
                      const Eigen::MatrixBase<LowRank> &U_star,      // (M, J)
                      const Eigen::MatrixBase<LowRank> &V_star,      // (M, J)
                      const Eigen::MatrixBase<Indices> &inds,        // (M)  ->  Index where the mth data point should be
                                                                     // inserted (the output of search_sorted)
                      Eigen::MatrixBase<RightHandSide> const &mu_out // (M)
) {
  int N = U.rows(), J = U.cols(), M = U_star.rows();
  CAST(RightHandSide, mu, M);
  Eigen::Matrix<typename LowRank::Scalar, 1, LowRank::ColsAtCompileTime> q(1, J);

  // Forward pass
  int m = 0;
  q.setZero();
  while (m < M && inds(m) <= 0) {
    mu(m) = 0;
    ++m;
  }
  for (int n = 0; n < N - 1; ++n) {
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
  for (int n = N - 1; n > 0; --n) {
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
