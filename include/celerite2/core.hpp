#ifndef _CELERITE2_CORE_HPP_DEFINED_
#define _CELERITE2_CORE_HPP_DEFINED_

#include <Eigen/Core>
#include <cassert>

namespace celerite {
namespace core {

template <typename a_t, typename U_t, typename P_t, typename K_t>
void to_dense(const Eigen::MatrixBase<a_t> &a, // (N)
              const Eigen::MatrixBase<U_t> &U, // (N, J)
              const Eigen::MatrixBase<U_t> &V, // (N, J)
              const Eigen::MatrixBase<P_t> &P, // (N-1, J)
              Eigen::MatrixBase<K_t> const &K_ // (N, N)
) {
  typedef typename Eigen::internal::plain_row_type<U_t>::type RowVector;

  int N = U.rows(), J = U.cols();

  Eigen::MatrixBase<K_t> &K = const_cast<Eigen::MatrixBase<K_t> &>(K_);
  K.derived().resize(N, N);

  RowVector un(1, J), vm(1, J), p(1, J);
  for (int m = 0; m < N; ++m) {
    vm = V.row(m);
    p.setConstant(1.0);
    K(m, m) = a(m);
    for (int n = m + 1; n < N; ++n) {
      p.array() *= P.row(n - 1).array();
      un      = U.row(n);
      K(n, m) = (un.array() * vm.array() * p.array()).sum();
      K(m, n) = K(n, m);
    }
  }
}

template <typename a_t, typename U_t, typename P_t, typename Z_t, typename Y_t>
void matmul(const Eigen::MatrixBase<a_t> &a, // (N)
            const Eigen::MatrixBase<U_t> &U, // (N, J)
            const Eigen::MatrixBase<U_t> &V, // (N, J)
            const Eigen::MatrixBase<P_t> &P, // (N-1, J)
            const Eigen::MatrixBase<Z_t> &Z, // (N, Nrhs)
            Eigen::MatrixBase<Y_t> const &Y_ // (N, Nrhs)
) {
  typedef typename a_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), Nrhs = Z.cols();

  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> F(J, Nrhs);
  Eigen::MatrixBase<Y_t> &Y = const_cast<Eigen::MatrixBase<Y_t> &>(Y_);
  Y.derived().resize(N, Nrhs);

  Y.row(N - 1) = a(N - 1) * Z.row(N - 1);

  F.setZero();
  for (int n = N - 2; n >= 0; --n) {
    F        = P.row(n).asDiagonal() * (F + U.row(n + 1).transpose() * Z.row(n + 1));
    Y.row(n) = a(n) * Z.row(n) + V.row(n) * F;
  }

  F.setZero();
  for (int n = 1; n < N; ++n) {
    F = P.row(n - 1).asDiagonal() * (F + V.row(n - 1).transpose() * Z.row(n - 1));
    Y.row(n) += U.row(n) * F;
  }
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename S_t>
int factor(const Eigen::MatrixBase<U_t> &U,  // (N, J)
           const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
           Eigen::MatrixBase<d_t> const &d_, // (N);    initially set to A
           Eigen::MatrixBase<W_t> const &W_, // (N, J); initially set to V
           Eigen::MatrixBase<S_t> const &S_  // (N, J*J)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp = U_t::ColsAtCompileTime;
  typedef typename Eigen::internal::plain_row_type<U_t>::type RowVector;

  int N = U.rows(), J = U.cols();

  RowVector tmp;
  Eigen::Matrix<Scalar, J_comp, J_comp> Sn(J, J);

  Eigen::MatrixBase<d_t> &d = const_cast<Eigen::MatrixBase<d_t> &>(d_);
  Eigen::MatrixBase<W_t> &W = const_cast<Eigen::MatrixBase<W_t> &>(W_);
  Eigen::MatrixBase<S_t> &S = const_cast<Eigen::MatrixBase<S_t> &>(S_);
  S.derived().resize(N, J * J);

  // First row
  Sn.setZero();
  S.row(0).setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    Sn.noalias() += d(n - 1) * W.row(n - 1).transpose() * W.row(n - 1);
    Sn = P.row(n - 1).asDiagonal() * Sn;
    // Sn.array() *= (P.row(n-1).transpose() * P.row(n-1)).array();
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k) S(n, j * J + k) = Sn(k, j);
    Sn *= P.row(n - 1).asDiagonal();

    // Update d = a - U * S * U.T
    tmp = U.row(n) * Sn;
    d(n) -= tmp * U.row(n).transpose();
    if (d(n) <= 0.0) return n;

    // Update W = (V - U * S) / d
    W.row(n).noalias() -= tmp;
    W.row(n) /= d(n);
  }

  return 0;
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename S_t, typename bU_t, typename bP_t, typename ba_t, typename bV_t>
void factor_grad(const Eigen::MatrixBase<U_t> &U,    // (N, J)
                 const Eigen::MatrixBase<P_t> &P,    // (N-1, J)
                 const Eigen::MatrixBase<d_t> &d,    // (N)
                 const Eigen::MatrixBase<W_t> &W,    // (N, J)
                 const Eigen::MatrixBase<S_t> &S,    // (N, J*J)
                 Eigen::MatrixBase<bU_t> const &bU_, // (N, J)
                 Eigen::MatrixBase<bP_t> const &bP_, // (N-1, J)
                 Eigen::MatrixBase<ba_t> const &ba_, // (N);    initially set to bd
                 Eigen::MatrixBase<bV_t> const &bV_  // (N, J); initially set to bW
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp = U_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols();

  // Make local copies of the gradients that we need.
  Eigen::Matrix<Scalar, J_comp, J_comp> Sn(J, J), bS(J, J);
  Eigen::Matrix<Scalar, J_comp, 1> bSWT;

  Eigen::MatrixBase<bU_t> &bU = const_cast<Eigen::MatrixBase<bU_t> &>(bU_);
  Eigen::MatrixBase<bP_t> &bP = const_cast<Eigen::MatrixBase<bP_t> &>(bP_);
  Eigen::MatrixBase<ba_t> &ba = const_cast<Eigen::MatrixBase<ba_t> &>(ba_);
  Eigen::MatrixBase<bV_t> &bV = const_cast<Eigen::MatrixBase<bV_t> &>(bV_);
  bU.derived().resize(N, J);
  bP.derived().resize(N - 1, J);

  bS.setZero();
  bV.array().colwise() /= d.array();
  for (int n = N - 1; n > 0; --n) {
    for (int j = 0; j < J; ++j)
      for (int k = 0; k < J; ++k) Sn(k, j) = S(n, j * J + k);

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

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t, typename F_t, typename G_t>
void solve(const Eigen::MatrixBase<U_t> &U,  // (N, J)
           const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
           const Eigen::MatrixBase<d_t> &d,  // (N)
           const Eigen::MatrixBase<W_t> &W,  // (N, J)
           Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs); initially set to Y
           Eigen::MatrixBase<F_t> const &F_, // (N, J*Nrhs)
           Eigen::MatrixBase<G_t> const &G_  // (N, J*Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), nrhs = Z_.cols();

  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs);

  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);
  Eigen::MatrixBase<F_t> &F = const_cast<Eigen::MatrixBase<F_t> &>(F_);
  Eigen::MatrixBase<G_t> &G = const_cast<Eigen::MatrixBase<G_t> &>(G_);
  F.derived().resize(N, J * nrhs);
  G.derived().resize(N, J * nrhs);

  Fn.setZero();
  F.row(0).setZero();

  for (int n = 1; n < N; ++n) {
    Fn.noalias() += W.row(n - 1).transpose() * Z.row(n - 1);
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) F(n, j * nrhs + k) = Fn(j, k);
    Fn = P.row(n - 1).asDiagonal() * Fn;
    Z.row(n).noalias() -= U.row(n) * Fn;
  }

  Z.array().colwise() /= d.array();

  Fn.setZero();
  G.row(N - 1).setZero();
  for (int n = N - 2; n >= 0; --n) {
    Fn.noalias() += U.row(n + 1).transpose() * Z.row(n + 1);
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) G(n, j * nrhs + k) = Fn(j, k);
    Fn = P.row(n).asDiagonal() * Fn;
    Z.row(n).noalias() -= W.row(n) * Fn;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
void solve_grad(const Eigen::MatrixBase<T1> &U,  // (N, J)
                const Eigen::MatrixBase<T2> &P,  // (N-1, J)
                const Eigen::MatrixBase<T3> &d,  // (N)
                const Eigen::MatrixBase<T1> &W,  // (N, J)
                const Eigen::MatrixBase<T4> &Z,  // (N, Nrhs)
                const Eigen::MatrixBase<T5> &F,  // (N, J*Nrhs)
                const Eigen::MatrixBase<T5> &G,  // (N, J*Nrhs)
                const Eigen::MatrixBase<T4> &bZ, // (N, Nrhs)
                Eigen::MatrixBase<T6> &bU,       // (N, J)
                Eigen::MatrixBase<T7> &bP,       // (N-1, J)
                Eigen::MatrixBase<T8> &bd,       // (N)
                Eigen::MatrixBase<T6> &bW,       // (N, J)
                Eigen::MatrixBase<T9> &bY        // (N, Nrhs)
) {
  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<typename T4::Scalar, T4::RowsAtCompileTime, T4::ColsAtCompileTime, T4::IsRowMajor> Z_ = Z;
  typedef Eigen::Matrix<typename T1::Scalar, T1::ColsAtCompileTime, Eigen::RowMajor> F_t;
  F_t F_(J, nrhs), bF = F_t::Zero(J, nrhs);

  bY = bZ;
  for (int n = 0; n <= N - 2; ++n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) F_(j, k) = G(n, j * nrhs + k);

    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() -= bY.row(n) * (P.row(n).asDiagonal() * F_).transpose();
    bF.noalias() -= W.row(n).transpose() * bY.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * (P.row(n).asDiagonal() * F_);

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() += (F_ * bF.transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n + 1).noalias() += Z_.row(n + 1) * bF.transpose();
    bY.row(n + 1).noalias() += U.row(n + 1) * bF;
  }

  bY.array().colwise() /= d.array();
  bd.array() -= (Z_.array() * bY.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  bF.setZero();
  for (int n = N - 1; n >= 1; --n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) F_(j, k) = F(n, j * nrhs + k);

    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bY.row(n) * (P.row(n - 1).asDiagonal() * F_).transpose();
    bF.noalias() -= U.row(n).transpose() * bY.row(n);

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n - 1).noalias() += (F_ * bF.transpose()).diagonal();
    bF = P.row(n - 1).asDiagonal() * bF;

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n - 1).noalias() += Z_.row(n - 1) * bF.transpose();
    bY.row(n - 1).noalias() += W.row(n - 1) * bF;
  }
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t>
void dot_l(const Eigen::MatrixBase<U_t> &U, // (N, J)
           const Eigen::MatrixBase<P_t> &P, // (N-1, J)
           const Eigen::MatrixBase<d_t> &d, // (N)
           const Eigen::MatrixBase<W_t> &W, // (N, J)
           Eigen::MatrixBase<Z_t> const &Z_ // (N, Nrhs); initially set to Y
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;
  typedef typename Eigen::internal::plain_row_type<Z_t>::type RowVectorZ;

  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);

  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<Scalar, d_t::RowsAtCompileTime, 1> sqrtd = sqrt(d.array());
  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> F(J, nrhs);

  F.setZero();
  Z.row(0) *= sqrtd(0);
  RowVectorZ tmp = Z.row(0);

  for (int n = 1; n < N; ++n) {
    F        = P.row(n - 1).asDiagonal() * (F + W.row(n - 1).transpose() * tmp);
    tmp      = sqrtd(n) * Z.row(n);
    Z.row(n) = tmp + U.row(n) * F;
  }
}

template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
void conditional_mean(const Eigen::MatrixBase<T1> &U,      // (N, J)
                      const Eigen::MatrixBase<T1> &V,      // (N, J)
                      const Eigen::MatrixBase<T2> &P,      // (N-1, J)
                      const Eigen::MatrixBase<T3> &z,      // (N)  ->  The result of a solve
                      const Eigen::MatrixBase<T4> &U_star, // (M, J)
                      const Eigen::MatrixBase<T4> &V_star, // (M, J)
                      const Eigen::MatrixBase<T5> &inds,   // (M)  ->  Index where the mth data point should be
                                                           // inserted (the output of search_sorted)
                      Eigen::MatrixBase<T6> &mu) {
  int N = U.rows(), J = U.cols(), M = U_star.rows();

  Eigen::Matrix<typename T1::Scalar, 1, T1::ColsAtCompileTime> q(1, J);

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
} // namespace celerite

#endif // _CELERITE2_CORE_HPP_DEFINED_
