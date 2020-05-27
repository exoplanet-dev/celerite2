#ifndef _CELERITE2_CORE_HPP_DEFINED_
#define _CELERITE2_CORE_HPP_DEFINED_

#include <Eigen/Core>

namespace celerite2 {
namespace core {

// ********
// INTERNAL
// ********

// Implementations of some of the forward ops live within an "internal" namespace to deal with optional arguments
namespace internal {

#ifdef __cpp_if_constexpr
#define if_constexpr if constexpr
#else
#define if_constexpr if
#endif

template <bool general, typename U_t, typename P_t, typename d_t, typename W_t, typename S_t>
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
  if_constexpr(general) {
    S.derived().resize(N, J * J);
    S.row(0).setZero();
  }

  // First row
  Sn.setZero();
  W.row(0) /= d(0);

  // The rest of the rows
  for (int n = 1; n < N; ++n) {
    // Update S = diag(P) * (S + d*W*W.T) * diag(P)
    Sn.noalias() += d(n - 1) * W.row(n - 1).transpose() * W.row(n - 1);
    Sn = P.row(n - 1).asDiagonal() * Sn;
    if_constexpr(general) {
      for (int j = 0; j < J; ++j)
        for (int k = 0; k < J; ++k) S(n, j * J + k) = Sn(k, j);
    }
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

template <bool is_solve, bool general, typename U_t, typename P_t, typename W_t, typename Y_t, typename Z_t, typename F_t>
void forward_sweep(const Eigen::MatrixBase<U_t> &U,  // (N, J)
                   const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
                   const Eigen::MatrixBase<W_t> &W,  // (N, J)
                   const Eigen::MatrixBase<Y_t> &Y,  // (N, Nrhs)
                   Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs)
                   Eigen::MatrixBase<F_t> const &F_  // (N, J*Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;
  typedef typename Eigen::internal::plain_row_type<Z_t>::type RowVectorZ;

  int N = U.rows(), J = U.cols(), nrhs = Z_.cols();
  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs);

  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);
  Eigen::MatrixBase<F_t> &F = const_cast<Eigen::MatrixBase<F_t> &>(F_);
  if_constexpr(general) {
    F.derived().resize(N, J * nrhs);
    F.row(0).setZero();
  }

  RowVectorZ tmp = Y.row(0); // This is required to handle inplace operations
  Fn.setZero();
  for (int n = 1; n < N; ++n) {
    Fn.noalias() += W.row(n - 1).transpose() * tmp;
    if_constexpr(general) {
      for (int k = 0; k < nrhs; ++k)
        for (int j = 0; j < J; ++j) F(n, k * J + j) = Fn(j, k);
    }
    Fn = P.row(n - 1).asDiagonal() * Fn;
    if_constexpr(is_solve) {
      Z.row(n).noalias() -= U.row(n) * Fn;
      tmp = Z.row(n);
    }
    else {
      tmp = Y.row(n);
      Z.row(n).noalias() += U.row(n) * Fn;
    }
  }
}

template <bool is_solve, bool general, typename U_t, typename P_t, typename W_t, typename Y_t, typename Z_t, typename G_t>
void backward_sweep(const Eigen::MatrixBase<U_t> &U,  // (N, J)
                    const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
                    const Eigen::MatrixBase<W_t> &W,  // (N, J)
                    const Eigen::MatrixBase<Y_t> &Y,  // (N, Nrhs)
                    Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs)
                    Eigen::MatrixBase<G_t> const &G_  // (N, J*Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;
  typedef typename Eigen::internal::plain_row_type<Z_t>::type RowVectorZ;

  int N = U.rows(), J = U.cols(), nrhs = Z_.cols();
  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs);

  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);
  Eigen::MatrixBase<G_t> &G = const_cast<Eigen::MatrixBase<G_t> &>(G_);
  if_constexpr(general) {
    G.derived().resize(N, J * nrhs);
    G.row(N - 1).setZero();
  }

  RowVectorZ tmp = Y.row(N - 1);
  Fn.setZero();
  for (int n = N - 2; n >= 0; --n) {
    Fn.noalias() += U.row(n + 1).transpose() * tmp;
    if_constexpr(general) {
      for (int k = 0; k < nrhs; ++k)
        for (int j = 0; j < J; ++j) G(n, k * J + j) = Fn(j, k);
    }
    Fn = P.row(n).asDiagonal() * Fn;
    if_constexpr(is_solve) {
      Z.row(n).noalias() -= W.row(n) * Fn;
      tmp = Z.row(n);
    }
    else {
      tmp = Y.row(n);
      Z.row(n).noalias() += W.row(n) * Fn;
    }
  }
}

template <bool is_solve, typename U_t, typename P_t, typename W_t, typename Y_t, typename Z_t, typename F_t, typename bZ_t, typename bU_t,
          typename bP_t, typename bW_t, typename bY_t>
void forward_sweep_grad(const Eigen::MatrixBase<U_t> &U,    // (N, J)
                        const Eigen::MatrixBase<P_t> &P,    // (N-1, J)
                        const Eigen::MatrixBase<W_t> &W,    // (N, J)
                        const Eigen::MatrixBase<Y_t> &Y,    // (N, Nrhs)
                        const Eigen::MatrixBase<Z_t> &Z,    // (N, Nrhs)
                        const Eigen::MatrixBase<F_t> &F,    // (N, J*Nrhs)
                        const Eigen::MatrixBase<bZ_t> &bZ,  // (N, Nrhs)
                        Eigen::MatrixBase<bU_t> const &bU_, // (N, J)
                        Eigen::MatrixBase<bP_t> const &bP_, // (N-1, J)
                        Eigen::MatrixBase<bW_t> const &bW_, // (N, J)
                        Eigen::MatrixBase<bY_t> const &bY_  // (N, Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), nrhs = Z.cols();
  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs), bF(J, nrhs);

  Eigen::MatrixBase<bU_t> &bU = const_cast<Eigen::MatrixBase<bU_t> &>(bU_);
  Eigen::MatrixBase<bP_t> &bP = const_cast<Eigen::MatrixBase<bP_t> &>(bP_);
  Eigen::MatrixBase<bW_t> &bW = const_cast<Eigen::MatrixBase<bW_t> &>(bW_);
  Eigen::MatrixBase<bY_t> &bY = const_cast<Eigen::MatrixBase<bY_t> &>(bY_);

  bF.setZero();
  for (int n = N - 1; n >= 1; --n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) Fn(j, k) = F(n, k * J + j);

    if_constexpr(is_solve) {
      // Grad of: Z.row(n).noalias() -= U.row(n) * Fn;
      bU.row(n).noalias() -= bZ.row(n) * (P.row(n - 1).asDiagonal() * Fn).transpose();
      bF.noalias() -= U.row(n).transpose() * bZ.row(n);
    }
    else {
      // Grad of: Z.row(n).noalias() += U.row(n) * Fn;
      bU.row(n).noalias() += bZ.row(n) * (P.row(n - 1).asDiagonal() * Fn).transpose();
      bF.noalias() += U.row(n).transpose() * bZ.row(n);
    }

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n - 1).noalias() += (Fn * bF.transpose()).diagonal();
    bF = P.row(n - 1).asDiagonal() * bF;

    if_constexpr(is_solve) {
      // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
      bW.row(n - 1).noalias() += Z.row(n - 1) * bF.transpose();
      bY.row(n - 1).noalias() += W.row(n - 1) * bF;
    }
    else {
      // Grad of: F.noalias() += W.row(n-1).transpose() * Y.row(n-1);
      bW.row(n - 1).noalias() += Y.row(n - 1) * bF.transpose();
      bY.row(n - 1).noalias() += W.row(n - 1) * bF;
    }
  }
}

template <bool is_solve, typename U_t, typename P_t, typename W_t, typename Y_t, typename Z_t, typename G_t, typename bZ_t, typename bU_t,
          typename bP_t, typename bW_t, typename bY_t>
void backward_sweep_grad(const Eigen::MatrixBase<U_t> &U,    // (N, J)
                         const Eigen::MatrixBase<P_t> &P,    // (N-1, J)
                         const Eigen::MatrixBase<W_t> &W,    // (N, J)
                         const Eigen::MatrixBase<Y_t> &Y,    // (N, Nrhs)
                         const Eigen::MatrixBase<Z_t> &Z,    // (N, Nrhs)
                         const Eigen::MatrixBase<G_t> &G,    // (N, J*Nrhs)
                         const Eigen::MatrixBase<bZ_t> &bZ,  // (N, Nrhs)
                         Eigen::MatrixBase<bU_t> const &bU_, // (N, J)
                         Eigen::MatrixBase<bP_t> const &bP_, // (N-1, J)
                         Eigen::MatrixBase<bW_t> const &bW_, // (N, J)
                         Eigen::MatrixBase<bY_t> const &bY_  // (N, Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), nrhs = Z.cols();
  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs), bF(J, nrhs);

  Eigen::MatrixBase<bU_t> &bU = const_cast<Eigen::MatrixBase<bU_t> &>(bU_);
  Eigen::MatrixBase<bP_t> &bP = const_cast<Eigen::MatrixBase<bP_t> &>(bP_);
  Eigen::MatrixBase<bW_t> &bW = const_cast<Eigen::MatrixBase<bW_t> &>(bW_);
  Eigen::MatrixBase<bY_t> &bY = const_cast<Eigen::MatrixBase<bY_t> &>(bY_);

  bF.setZero();
  for (int n = 0; n <= N - 2; ++n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) Fn(j, k) = G(n, k * J + j);

    if_constexpr(is_solve) {
      // Grad of: Z.row(n).noalias() -= W.row(n) * Fn;
      bW.row(n).noalias() -= bZ.row(n) * (P.row(n).asDiagonal() * Fn).transpose();
      bF.noalias() -= W.row(n).transpose() * bZ.row(n);
    }
    else {
      // Grad of: Z.row(n).noalias() += W.row(n) * Fn;
      bW.row(n).noalias() += bZ.row(n) * (P.row(n).asDiagonal() * Fn).transpose();
      bF.noalias() += W.row(n).transpose() * bZ.row(n);
    }

    // Grad of: F = P.row(n).asDiagonal() * F;
    bP.row(n).noalias() += (Fn * bF.transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    if_constexpr(is_solve) {
      // Grad of: F.noalias() += U.row(n+1).transpose() * Z.row(n+1);
      bU.row(n + 1).noalias() += Z.row(n + 1) * bF.transpose();
      bY.row(n + 1).noalias() += U.row(n + 1) * bF;
    }
    else {
      // Grad of: F.noalias() += U.row(n+1).transpose() * Y.row(n+1);
      bU.row(n + 1).noalias() += Y.row(n + 1) * bF.transpose();
      bY.row(n + 1).noalias() += U.row(n + 1) * bF;
    }
  }
}

template <bool general, typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t, typename F_t, typename G_t>
void solve(const Eigen::MatrixBase<U_t> &U,  // (N, J)
           const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
           const Eigen::MatrixBase<d_t> &d,  // (N)
           const Eigen::MatrixBase<W_t> &W,  // (N, J)
           Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs); initially set to Y
           Eigen::MatrixBase<F_t> const &F_, // (N, J*Nrhs)
           Eigen::MatrixBase<G_t> const &G_  // (N, J*Nrhs)
) {
  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);
  forward_sweep<true, general, U_t, P_t, W_t, Z_t, Z_t, F_t>(U, P, W, Z, Z, F_);
  Z.array().colwise() /= d.array();
  backward_sweep<true, general, U_t, P_t, W_t, Z_t, Z_t, G_t>(U, P, W, Z, Z, G_);
}

template <bool general, typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t, typename F_t>
void dot_tril(const Eigen::MatrixBase<U_t> &U,  // (N, J)
              const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
              const Eigen::MatrixBase<d_t> &d,  // (N)
              const Eigen::MatrixBase<W_t> &W,  // (N, J)
              Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs); initially set to Y
              Eigen::MatrixBase<F_t> const &F_  // (N, J*Nrhs)
) {

  Eigen::MatrixBase<Z_t> &Z = const_cast<Eigen::MatrixBase<Z_t> &>(Z_);
  Z.array().colwise() *= sqrt(d.array());
  forward_sweep<false, general, U_t, P_t, W_t, Z_t, Z_t, F_t>(U, P, W, Z, Z, F_);
}

template <bool general, typename a_t, typename U_t, typename V_t, typename P_t, typename Z_t, typename Y_t, typename F_t, typename G_t>
void matmul(const Eigen::MatrixBase<a_t> &a,  // (N)
            const Eigen::MatrixBase<U_t> &U,  // (N, J)
            const Eigen::MatrixBase<V_t> &V,  // (N, J)
            const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
            const Eigen::MatrixBase<Z_t> &Z,  // (N, Nrhs)
            Eigen::MatrixBase<Y_t> const &Y_, // (N, Nrhs)
            Eigen::MatrixBase<F_t> const &F_, // (N, J*Nrhs)
            Eigen::MatrixBase<G_t> const &G_  // (N, J*Nrhs)
) {
  Eigen::MatrixBase<Y_t> &Y = const_cast<Eigen::MatrixBase<Y_t> &>(Y_);
  Y                         = Z;
  Y.array().colwise() *= a.array();
  forward_sweep<false, general, U_t, P_t, V_t, Z_t, Y_t, F_t>(U, P, V, Z, Y, F_);
  backward_sweep<false, general, U_t, P_t, V_t, Z_t, Y_t, F_t>(U, P, V, Z, Y, G_);
}

#undef if_constexpr

} // namespace internal

// ****************
// PUBLIC INTERFACE
// ****************

#define MakeEmpty                                                                                                                                    \
  typedef typename U_t::Scalar Scalar;                                                                                                               \
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Empty;                                                                               \
  Empty

template <typename U_t, typename P_t, typename d_t, typename W_t, typename S_t>
int factor(const Eigen::MatrixBase<U_t> &U,  // (N, J)
           const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
           Eigen::MatrixBase<d_t> const &d_, // (N);    initially set to A
           Eigen::MatrixBase<W_t> const &W_, // (N, J); initially set to V
           Eigen::MatrixBase<S_t> const &S_  // (N, J*J)
) {
  return internal::factor<true, U_t, P_t, d_t, W_t, S_t>(U, P, d_, W_, S_);
}

template <typename U_t, typename P_t, typename d_t, typename W_t>
int factor(const Eigen::MatrixBase<U_t> &U,  // (N, J)
           const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
           Eigen::MatrixBase<d_t> const &d_, // (N);    initially set to A
           Eigen::MatrixBase<W_t> const &W_  // (N, J); initially set to V
) {
  MakeEmpty S;
  return internal::factor<false, U_t, P_t, d_t, W_t, Empty>(U, P, d_, W_, S);
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
  return internal::solve<true, U_t, P_t, d_t, W_t, Z_t, F_t, G_t>(U, P, d, W, Z_, F_, G_);
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t>
void solve(const Eigen::MatrixBase<U_t> &U, // (N, J)
           const Eigen::MatrixBase<P_t> &P, // (N-1, J)
           const Eigen::MatrixBase<d_t> &d, // (N)
           const Eigen::MatrixBase<W_t> &W, // (N, J)
           Eigen::MatrixBase<Z_t> const &Z_ // (N, Nrhs); initially set to Y
) {
  MakeEmpty F, G;
  return internal::solve<false, U_t, P_t, d_t, W_t, Z_t, Empty, Empty>(U, P, d, W, Z_, F, G);
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t>
void dot_tril(const Eigen::MatrixBase<U_t> &U, // (N, J)
              const Eigen::MatrixBase<P_t> &P, // (N-1, J)
              const Eigen::MatrixBase<d_t> &d, // (N)
              const Eigen::MatrixBase<W_t> &W, // (N, J)
              Eigen::MatrixBase<Z_t> const &Z_ // (N, Nrhs); initially set to Y
) {
  MakeEmpty F;
  return internal::dot_tril<false, U_t, P_t, d_t, W_t, Z_t, Empty>(U, P, d, W, Z_, F);
}

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t, typename F_t>
void dot_tril(const Eigen::MatrixBase<U_t> &U,  // (N, J)
              const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
              const Eigen::MatrixBase<d_t> &d,  // (N)
              const Eigen::MatrixBase<W_t> &W,  // (N, J)
              Eigen::MatrixBase<Z_t> const &Z_, // (N, Nrhs); initially set to Y
              Eigen::MatrixBase<F_t> const &F_  // (N, J*Nrhs)
) {
  return internal::dot_tril<true, U_t, P_t, d_t, W_t, Z_t, F_t>(U, P, d, W, Z_, F_);
}

template <typename a_t, typename U_t, typename V_t, typename P_t, typename Z_t, typename Y_t, typename F_t, typename G_t>
void matmul(const Eigen::MatrixBase<a_t> &a,  // (N)
            const Eigen::MatrixBase<U_t> &U,  // (N, J)
            const Eigen::MatrixBase<V_t> &V,  // (N, J)
            const Eigen::MatrixBase<P_t> &P,  // (N-1, J)
            const Eigen::MatrixBase<Z_t> &Z,  // (N, Nrhs)
            Eigen::MatrixBase<Y_t> const &Y_, // (N, Nrhs)
            Eigen::MatrixBase<F_t> const &F_, // (N, J*Nrhs)
            Eigen::MatrixBase<G_t> const &G_  // (N, J*Nrhs)
) {
  return internal::matmul<true, a_t, U_t, V_t, P_t, Z_t, Y_t, F_t, G_t>(a, U, V, P, Z, Y_, F_, G_);
}

template <typename a_t, typename U_t, typename V_t, typename P_t, typename Z_t, typename Y_t>
void matmul(const Eigen::MatrixBase<a_t> &a, // (N)
            const Eigen::MatrixBase<U_t> &U, // (N, J)
            const Eigen::MatrixBase<V_t> &V, // (N, J)
            const Eigen::MatrixBase<P_t> &P, // (N-1, J)
            const Eigen::MatrixBase<Z_t> &Z, // (N, Nrhs)
            Eigen::MatrixBase<Y_t> const &Y_ // (N, Nrhs)
) {
  MakeEmpty F, G;
  return internal::matmul<false, a_t, U_t, V_t, P_t, Z_t, Y_t, Empty, Empty>(a, U, V, P, Z, Y_, F, G);
}

#undef MakeEmpty

// ********
// BACKPROP
// ********

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

template <typename U_t, typename P_t, typename d_t, typename W_t, typename Z_t, typename F_t, typename G_t, typename bZ_t, typename bU_t,
          typename bP_t, typename bd_t, typename bW_t, typename bY_t>
void solve_grad(const Eigen::MatrixBase<U_t> &U,    // (N, J)
                const Eigen::MatrixBase<P_t> &P,    // (N-1, J)
                const Eigen::MatrixBase<d_t> &d,    // (N)
                const Eigen::MatrixBase<W_t> &W,    // (N, J)
                const Eigen::MatrixBase<Z_t> &Z,    // (N, Nrhs)
                const Eigen::MatrixBase<F_t> &F,    // (N, J*Nrhs)
                const Eigen::MatrixBase<G_t> &G,    // (N, J*Nrhs)
                const Eigen::MatrixBase<bZ_t> &bZ,  // (N, Nrhs)
                Eigen::MatrixBase<bU_t> const &bU_, // (N, J)
                Eigen::MatrixBase<bP_t> const &bP_, // (N-1, J)
                Eigen::MatrixBase<bd_t> const &bd_, // (N)
                Eigen::MatrixBase<bW_t> const &bW_, // (N, J)
                Eigen::MatrixBase<bY_t> const &bY_  // (N, Nrhs)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp    = U_t::ColsAtCompileTime;
  constexpr int Nrhs_comp = Z_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), nrhs = Z.cols();

  Eigen::Matrix<Scalar, Z_t::RowsAtCompileTime, Z_t::ColsAtCompileTime, Z_t::IsRowMajor> Z_ = Z;

  Eigen::MatrixBase<bU_t> &bU = const_cast<Eigen::MatrixBase<bU_t> &>(bU_);
  Eigen::MatrixBase<bP_t> &bP = const_cast<Eigen::MatrixBase<bP_t> &>(bP_);
  Eigen::MatrixBase<bd_t> &bd = const_cast<Eigen::MatrixBase<bd_t> &>(bd_);
  Eigen::MatrixBase<bW_t> &bW = const_cast<Eigen::MatrixBase<bW_t> &>(bW_);
  Eigen::MatrixBase<bY_t> &bY = const_cast<Eigen::MatrixBase<bY_t> &>(bY_);
  bU.derived().resize(N, J);
  bP.derived().resize(N - 1, J);
  bd.derived().resize(N);
  bW.derived().resize(N, J);
  bY.derived().resize(N, nrhs);

  Eigen::Matrix<Scalar, J_comp, Nrhs_comp> Fn(J, nrhs), bF(J, nrhs);
  bF.setZero();

  bU.row(0).setZero();

  bY = bZ;
  for (int n = 0; n <= N - 2; ++n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) Fn(j, k) = G(n, k * J + j);

    // Grad of: Z.row(n).noalias() -= W.row(n) * G;
    bW.row(n).noalias() = -bY.row(n) * (P.row(n).asDiagonal() * Fn).transpose();
    bF.noalias() -= W.row(n).transpose() * bY.row(n);

    // Inverse of: Z.row(n).noalias() -= W.row(n) * G;
    Z_.row(n).noalias() += W.row(n) * (P.row(n).asDiagonal() * Fn);

    // Grad of: g = P.row(n).asDiagonal() * G;
    bP.row(n).noalias() = (Fn * bF.transpose()).diagonal();
    bF                  = P.row(n).asDiagonal() * bF;

    // Grad of: g.noalias() += U.row(n+1).transpose() * Z.row(n+1);
    bU.row(n + 1).noalias() = Z_.row(n + 1) * bF.transpose();
    bY.row(n + 1).noalias() += U.row(n + 1) * bF;
  }

  bW.row(N - 1).setZero();

  bY.array().colwise() /= d.array();
  bd.array() = -(Z_.array() * bY.array()).rowwise().sum();

  // Inverse of: Z.array().colwise() /= d.array();
  Z_.array().colwise() *= d.array();

  bF.setZero();
  for (int n = N - 1; n >= 1; --n) {
    for (int k = 0; k < nrhs; ++k)
      for (int j = 0; j < J; ++j) Fn(j, k) = F(n, k * J + j);

    // Grad of: Z.row(n).noalias() -= U.row(n) * f;
    bU.row(n).noalias() -= bY.row(n) * (P.row(n - 1).asDiagonal() * Fn).transpose();
    bF.noalias() -= U.row(n).transpose() * bY.row(n);

    // Grad of: F = P.row(n-1).asDiagonal() * F;
    bP.row(n - 1).noalias() += (Fn * bF.transpose()).diagonal();
    bF = P.row(n - 1).asDiagonal() * bF;

    // Grad of: F.noalias() += W.row(n-1).transpose() * Z.row(n-1);
    bW.row(n - 1).noalias() += Z_.row(n - 1) * bF.transpose();
    bY.row(n - 1).noalias() += W.row(n - 1) * bF;
  }

  bY -= bZ;
}

// ****************
// HELPER FUNCTIONS
// ****************

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

template <typename U_t, typename V_t, typename P_t, typename z_t, typename U_star_t, typename V_star_t, typename inds_t, typename mu_t>
void conditional_mean(const Eigen::MatrixBase<U_t> &U,           // (N, J)
                      const Eigen::MatrixBase<V_t> &V,           // (N, J)
                      const Eigen::MatrixBase<P_t> &P,           // (N-1, J)
                      const Eigen::MatrixBase<z_t> &z,           // (N)  ->  The result of a solve
                      const Eigen::MatrixBase<U_star_t> &U_star, // (M, J)
                      const Eigen::MatrixBase<V_star_t> &V_star, // (M, J)
                      const Eigen::MatrixBase<inds_t> &inds,     // (M)  ->  Index where the mth data point should be
                                                                 // inserted (the output of search_sorted)
                      Eigen::MatrixBase<mu_t> const &mu_         // (M)
) {
  typedef typename U_t::Scalar Scalar;
  constexpr int J_comp = U_t::ColsAtCompileTime;

  int N = U.rows(), J = U.cols(), M = U_star.rows();

  Eigen::Matrix<Scalar, 1, J_comp> q(1, J);
  Eigen::MatrixBase<mu_t> &mu = const_cast<Eigen::MatrixBase<mu_t> &>(mu_);
  mu.derived().resize(M);

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

#endif // _CELERITE2_CORE_HPP_DEFINED_
