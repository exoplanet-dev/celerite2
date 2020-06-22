#ifndef _CELERITE2_CORE2_HPP_DEFINED_
#define _CELERITE2_CORE2_HPP_DEFINED_

#include <Eigen/Core>

namespace celerite2 {
namespace core2 {

#define CAST_BASE(TYPE, VAR) Eigen::MatrixBase<TYPE> &VAR = const_cast<Eigen::MatrixBase<TYPE> &>(VAR##_in)

#define CAST_VEC(TYPE, VAR, ROWS)                                                                                                                    \
  CAST_BASE(TYPE, VAR);                                                                                                                              \
  VAR.derived().resize(ROWS)

#define CAST_MAT(TYPE, VAR, ROWS, COLS)                                                                                                              \
  CAST_BASE(TYPE, VAR);                                                                                                                              \
  VAR.derived().resize(ROWS, COLS)

#define GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define CAST(...) GET_MACRO(__VA_ARGS__, CAST_MAT, CAST_VEC, CAST_BASE)(__VA_ARGS__)

const int THE_WORKSPACE_VARIABLE_MUST_BE_ROW_MAJOR = 0;
#define ASSERT_ROW_MAJOR(TYPE) EIGEN_STATIC_ASSERT(TYPE::IsRowMajor, THE_WORKSPACE_VARIABLE_MUST_BE_ROW_MAJOR)

template <typename Diag, typename LowRank, typename Dense>
void to_dense(const Eigen::MatrixBase<Diag> &a,    // (N,)
              const Eigen::MatrixBase<LowRank> &U, // (N, J)
              const Eigen::MatrixBase<LowRank> &V, // (N, J)
              const Eigen::MatrixBase<LowRank> &P, // (N-1, J)
              Eigen::MatrixBase<Dense> const &K_in // (N,)
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

template <typename Diag, typename LowRank, typename Work>
int factor(const Eigen::MatrixBase<Diag> &a,       // (N,)
           const Eigen::MatrixBase<LowRank> &U,    // (N, J)
           const Eigen::MatrixBase<LowRank> &V,    // (N, J)
           const Eigen::MatrixBase<LowRank> &P,    // (N-1, J)
           Eigen::MatrixBase<Diag> const &d_in,    // (N,)
           Eigen::MatrixBase<LowRank> const &W_in, // (N, J)
           Eigen::MatrixBase<Work> const &S_in     // (N, J*J)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename Diag::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<LowRank>::type RowVector;

  int N = U.rows(), J = U.cols();
  CAST(Diag, d, N);
  CAST(LowRank, W, N, J);
  CAST(Work, S, N, J * J);

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
    S.row(n) = ptr;

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

template <typename Diag, typename LowRank, typename Work>
void factor_rev(const Eigen::MatrixBase<Diag> &a,        // (N,)
                const Eigen::MatrixBase<LowRank> &U,     // (N, J)
                const Eigen::MatrixBase<LowRank> &V,     // (N, J)
                const Eigen::MatrixBase<LowRank> &P,     // (N-1, J)
                const Eigen::MatrixBase<Diag> &d,        // (N,)
                const Eigen::MatrixBase<LowRank> &W,     // (N, J)
                const Eigen::MatrixBase<Work> &S,        // (N, J*J)
                const Eigen::MatrixBase<Diag> &bd,       // (N,)
                const Eigen::MatrixBase<LowRank> &bW,    // (N, J)
                Eigen::MatrixBase<Diag> const &ba_in,    // (N,)
                Eigen::MatrixBase<LowRank> const &bU_in, // (N, J)
                Eigen::MatrixBase<LowRank> const &bV_in, // (N, J)
                Eigen::MatrixBase<LowRank> const &bP_in  // (N-1, J)

) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename Diag::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, LowRank::ColsAtCompileTime> Inner;

  int N = U.rows(), J = U.cols();
  CAST(Diag, ba, N);
  CAST(LowRank, bU, N, J);
  CAST(LowRank, bV, N, J);
  CAST(LowRank, bP, N - 1, J);

  // Make local copies of the gradients that we need
  Inner Sn(J, J), bS(J, J);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Sn.data(), 1, J * J);
  Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, 1> bSWT;

  bS.setZero();
  ba.noalias() = bd;
  bV.noalias() = bW;
  bV.array().colwise() /= d.array();
  for (int n = N - 1; n > 0; --n) {
    ptr = S.row(n);

    // Step 6
    ba(n) -= W.row(n) * bV.row(n).transpose();
    bU.row(n).noalias() = -(bV.row(n) + 2.0 * ba(n) * U.row(n)) * Sn * P.row(n - 1).asDiagonal();
    bS.noalias() -= U.row(n).transpose() * (bV.row(n) + ba(n) * U.row(n));

    // Step 4
    bP.row(n - 1).noalias() = (bS * Sn + Sn.transpose() * bS).diagonal();

    // Step 3
    bS   = P.row(n - 1).asDiagonal() * bS * P.row(n - 1).asDiagonal();
    bSWT = bS * W.row(n - 1).transpose();
    ba(n - 1) += W.row(n - 1) * bSWT;
    bV.row(n - 1).noalias() += W.row(n - 1) * (bS + bS.transpose());
  }

  bU.row(0).setZero();
  ba(0) -= bV.row(0) * W.row(0).transpose();
}

namespace internal {

template <bool is_solve = false>
struct update_f {
  template <typename A, typename B, typename C, typename D>
  static void apply(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, Eigen::MatrixBase<D> const &d_in) {
    CAST(D, d);
    d.noalias() += a * b;
  }
};

template <>
struct update_f<true> {
  template <typename A, typename B, typename C, typename D>
  static void apply(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, Eigen::MatrixBase<D> const &d_in) {
    CAST(D, d);
    d.noalias() += a * c;
  }
};

template <bool is_solve = false>
struct update_z {
  template <typename A, typename B>
  static void apply(const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_in) {
    CAST(B, b);
    b.noalias() += a;
  }
};

template <>
struct update_z<true> {
  template <typename A, typename B>
  static void apply(const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_in) {
    CAST(B, b);
    b.noalias() -= a;
  }
};

template <bool is_solve, typename LowRank, typename RightHandSide, typename Work>
void forward(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
             const Eigen::MatrixBase<LowRank> &V,          // (N, J)
             const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
             const Eigen::MatrixBase<RightHandSide> &Y,    // (N, Nrhs)
             Eigen::MatrixBase<RightHandSide> const &Z_in, // (N, Nrhs)
             Eigen::MatrixBase<Work> const &F_in           // (N, J * Nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  int N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST(RightHandSide, Z); // Must already be the right shape
  CAST(Work, F, N, J * nrhs);
  F.row(0).setZero();

  Inner Fn(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);

  Fn.setZero();
  for (int n = 1; n < N; ++n) {
    update_f<is_solve>::apply(V.row(n - 1).transpose(), Y.row(n - 1), Z.row(n - 1), Fn);
    F.row(n) = ptr;
    Fn       = P.row(n - 1).asDiagonal() * Fn;
    update_z<is_solve>::apply(U.row(n) * Fn, Z.row(n));
  }
}

template <bool is_solve, typename LowRank, typename RightHandSide, typename Work>
void backward(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
              const Eigen::MatrixBase<LowRank> &V,          // (N, J)
              const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
              const Eigen::MatrixBase<RightHandSide> &Y,    // (N, Nrhs)
              Eigen::MatrixBase<RightHandSide> const &Z_in, // (N, Nrhs)
              Eigen::MatrixBase<Work> const &F_in           // (N, J * Nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  int N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST(RightHandSide, Z); // Must already be the right shape
  CAST(Work, F, N, J * nrhs);
  F.row(N - 1).setZero();

  Inner Fn(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);

  Fn.setZero();
  for (int n = N - 2; n >= 0; --n) {
    update_f<is_solve>::apply(U.row(n + 1).transpose(), Y.row(n + 1), Z.row(n + 1), Fn);
    F.row(n) = ptr;
    Fn       = P.row(n).asDiagonal() * Fn;
    update_z<is_solve>::apply(V.row(n) * Fn, Z.row(n));
  }
}

} // namespace internal

template <typename Diag, typename LowRank, typename RightHandSide, typename Work>
void solve(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
           const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
           const Eigen::MatrixBase<Diag> &d,             // (N,)
           const Eigen::MatrixBase<LowRank> &W,          // (N, J)
           const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
           Eigen::MatrixBase<RightHandSide> const &X_in, // (N, nrhs)
           Eigen::MatrixBase<RightHandSide> const &Z_in, // (N, nrhs)
           Eigen::MatrixBase<Work> const &F_in,          // (N, J*nrhs)
           Eigen::MatrixBase<Work> const &G_in           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, X);
  CAST(RightHandSide, Z);

  Z = Y;
  internal::forward<true>(U, W, P, Y, Z, F_in);

  X = Z;
  X.array().colwise() /= d.array();
  internal::backward<true>(U, W, P, Z, X, G_in);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename Work>
void dot_tril(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
              const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
              const Eigen::MatrixBase<Diag> &d,             // (N,)
              const Eigen::MatrixBase<LowRank> &W,          // (N, J)
              const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
              Eigen::MatrixBase<RightHandSide> const &Z_in, // (N, nrhs)
              Eigen::MatrixBase<Work> const &F_in           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, Z);

  // First dot in sqrt(d)
  RightHandSide tmp = Y;
  tmp.array().colwise() *= sqrt(d.array());

  // Then apply L
  Z = tmp;
  internal::forward<false>(U, W, P, tmp, Z, F_in);
}

//
template <typename Diag, typename LowRank, typename RightHandSide, typename Work>
void matmul(const Eigen::MatrixBase<Diag> &a,             // (N,)
            const Eigen::MatrixBase<LowRank> &U,          // (N, J)
            const Eigen::MatrixBase<LowRank> &V,          // (N, J)
            const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
            const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
            Eigen::MatrixBase<RightHandSide> const &X_in, // (N, nrhs)
            Eigen::MatrixBase<RightHandSide> const &Z_in, // (N, nrhs)
            Eigen::MatrixBase<Work> const &F_in,          // (N, J*nrhs)
            Eigen::MatrixBase<Work> const &G_in           // (N, J*nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, X);
  CAST(RightHandSide, Z);

  // Z = diag(a) * Y + tril(U V^T) * Y
  Z = a.asDiagonal() * Y;
  internal::forward<false>(U, V, P, Y, Z, F_in);

  // X = Z + triu(V U^T) * Y
  X = Z;
  internal::backward<false>(U, V, P, Y, X, G_in);
}

} // namespace core2
} // namespace celerite2

#endif // _CELERITE2_CORE2_HPP_DEFINED_
