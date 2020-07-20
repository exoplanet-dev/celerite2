#ifndef _CELERITE2_INTERNAL_HPP_DEFINED_
#define _CELERITE2_INTERNAL_HPP_DEFINED_

#include <Eigen/Core>

namespace celerite2 {
namespace core {

#define UNUSED(x) (void)(x)

#define CAST_BASE(TYPE, VAR) Eigen::MatrixBase<TYPE> &VAR = const_cast<Eigen::MatrixBase<TYPE> &>(VAR##_out)

#define CAST_VEC(TYPE, VAR, ROWS)                                                                                                                    \
  CAST_BASE(TYPE, VAR);                                                                                                                              \
  VAR.derived().resize(ROWS)

#define CAST_MAT(TYPE, VAR, ROWS, COLS)                                                                                                              \
  CAST_BASE(TYPE, VAR);                                                                                                                              \
  VAR.derived().resize(ROWS, COLS)

const int THE_WORKSPACE_VARIABLE_MUST_BE_ROW_MAJOR = 0;
#define ASSERT_ROW_MAJOR(TYPE) EIGEN_STATIC_ASSERT((TYPE::ColsAtCompileTime == 1) || TYPE::IsRowMajor, THE_WORKSPACE_VARIABLE_MUST_BE_ROW_MAJOR)

namespace internal {

template <bool do_update = true>
struct update_workspace {
  template <typename A, typename B>
  static void apply(Eigen::Index n, const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_out) {
    CAST_BASE(B, b);
    b.row(n) = a;
  }
};

template <>
struct update_workspace<false> {
  template <typename A, typename B>
  static void apply(Eigen::Index n, const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_out) {
    UNUSED(n);
    UNUSED(a);
    UNUSED(b_out);
  }
};

template <bool is_solve = false>
struct update_f {
  template <typename A, typename B, typename C, typename D>
  static void apply(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, Eigen::MatrixBase<D> const &d_out) {
    CAST_BASE(D, d);
    d.noalias() += a * b;
    UNUSED(c);
  }

  template <typename A, typename B, typename C, typename D, typename E, typename F, typename G>
  static void reverse(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, const Eigen::MatrixBase<D> &d,
                      Eigen::MatrixBase<E> const &e_out, Eigen::MatrixBase<F> const &f_out, Eigen::MatrixBase<G> const &g_out) {
    CAST_BASE(E, e);
    CAST_BASE(F, f);
    e.noalias() += b * d.transpose();
    f.noalias() += a * d;
    UNUSED(c);
    UNUSED(g_out);
  }
};

template <>
struct update_f<true> {
  template <typename A, typename B, typename C, typename D>
  static void apply(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, Eigen::MatrixBase<D> const &d_out) {
    CAST_BASE(D, d);
    d.noalias() += a * c;
    UNUSED(b);
  }

  template <typename A, typename B, typename C, typename D, typename E, typename F, typename G>
  static void reverse(const Eigen::MatrixBase<A> &a, const Eigen::MatrixBase<B> &b, const Eigen::MatrixBase<C> &c, const Eigen::MatrixBase<D> &d,
                      Eigen::MatrixBase<E> const &e_out, Eigen::MatrixBase<F> const &f_out, Eigen::MatrixBase<G> const &g_out) {
    CAST_BASE(E, e);
    CAST_BASE(G, g);
    e.noalias() += c * d.transpose();
    g.noalias() += a * d;
    UNUSED(b);
    UNUSED(f_out);
  }
};

template <bool is_solve = false>
struct update_z {
  template <typename A, typename B>
  static void apply(const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_out) {
    CAST_BASE(B, b);
    b.noalias() += a;
  }
};

template <>
struct update_z<true> {
  template <typename A, typename B>
  static void apply(const Eigen::MatrixBase<A> &a, Eigen::MatrixBase<B> const &b_out) {
    CAST_BASE(B, b);
    b.noalias() -= a;
  }
};

template <bool is_solve, bool do_update = true, typename LowRank, typename RightHandSide, typename RightHandSideOut, typename Work>
void forward(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
             const Eigen::MatrixBase<LowRank> &V,              // (N, J)
             const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
             const Eigen::MatrixBase<RightHandSide> &Y,        // (N, Nrhs)
             Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, Nrhs)
             Eigen::MatrixBase<Work> const &F_out              // (N, J * Nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<RightHandSide>::type RowVector;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST_BASE(RightHandSideOut, Z); // Must already be the right shape
  CAST_BASE(Work, F);
  if (do_update) {
    F.derived().resize(N, J * nrhs);
    F.row(0).setZero();
  }

  Inner Fn(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);

  // This will track the previous row allowing for inplace operations
  RowVector tmp = Y.row(0);

  Fn.setZero();
  for (Eigen::Index n = 1; n < N; ++n) {
    update_f<is_solve>::apply(V.row(n - 1).transpose(), tmp, Z.row(n - 1), Fn);
    tmp = Y.row(n);
    update_workspace<do_update>::apply(n, ptr, F);
    Fn = P.row(n - 1).asDiagonal() * Fn;
    update_z<is_solve>::apply(U.row(n) * Fn, Z.row(n));
  }
}

template <bool is_solve, bool do_update = true, typename LowRank, typename RightHandSide, typename RightHandSideOut, typename Work>
void backward(const Eigen::MatrixBase<LowRank> &U,              // (N, J)
              const Eigen::MatrixBase<LowRank> &V,              // (N, J)
              const Eigen::MatrixBase<LowRank> &P,              // (N-1, J)
              const Eigen::MatrixBase<RightHandSide> &Y,        // (N, Nrhs)
              Eigen::MatrixBase<RightHandSideOut> const &Z_out, // (N, Nrhs)
              Eigen::MatrixBase<Work> const &F_out              // (N, J * Nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::internal::plain_row_type<RightHandSide>::type RowVector;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST_BASE(RightHandSideOut, Z); // Must already be the right shape
  CAST_BASE(Work, F);
  if (do_update) {
    F.derived().resize(N, J * nrhs);
    F.row(N - 1).setZero();
  }

  Inner Fn(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);

  // This will track the previous row allowing for inplace operations
  RowVector tmp = Y.row(N - 1);

  Fn.setZero();
  for (Eigen::Index n = N - 2; n >= 0; --n) {
    update_f<is_solve>::apply(U.row(n + 1).transpose(), tmp, Z.row(n + 1), Fn);
    tmp = Y.row(n);
    update_workspace<do_update>::apply(n, ptr, F);
    Fn = P.row(n).asDiagonal() * Fn;
    update_z<is_solve>::apply(V.row(n) * Fn, Z.row(n));
  }
}

template <bool is_solve, typename LowRank, typename RightHandSide, typename RightHandSideOptional, typename Work, typename RightHandSideInternal,
          typename LowRankOut,
          typename RightHandSideOut>
void forward_rev(const Eigen::MatrixBase<LowRank> &U,                    // (N, J)
                 const Eigen::MatrixBase<LowRank> &V,                    // (N, J)
                 const Eigen::MatrixBase<LowRank> &P,                    // (N-1, J)
                 const Eigen::MatrixBase<RightHandSideOptional> &Y,      // (N, Nrhs)
                 const Eigen::MatrixBase<RightHandSide> &Z,              // (N, Nrhs)
                 const Eigen::MatrixBase<Work> &F,                       // (N, J * Nrhs)
                 Eigen::MatrixBase<RightHandSideInternal> const &bZ_out, // (N, Nrhs)
                 Eigen::MatrixBase<LowRankOut> const &bU_out,            // (N, J)
                 Eigen::MatrixBase<LowRankOut> const &bV_out,            // (N, J)
                 Eigen::MatrixBase<LowRankOut> const &bP_out,            // (N-1, J)
                 Eigen::MatrixBase<RightHandSideOut> const &bY_out       // (N, Nrhs)  -  Must be the right shape already (and zeroed)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST_MAT(LowRankOut, bU, N, J);
  CAST_MAT(LowRankOut, bV, N, J);
  CAST_MAT(LowRankOut, bP, N - 1, J);
  CAST_BASE(RightHandSideOut, bY);
  CAST_BASE(RightHandSideInternal, bZ);

  Inner Fn(J, nrhs), bF(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);
  bF.setZero();
  for (Eigen::Index n = N - 1; n >= 1; --n) {
    ptr = F.row(n);

    // Reverse: update_z<is_solve>::apply(U.row(n) * Fn, Z.row(n));
    update_z<is_solve>::apply(bZ.row(n) * (P.row(n - 1).asDiagonal() * Fn).transpose(), bU.row(n));
    update_z<is_solve>::apply(U.row(n).transpose() * bZ.row(n), bF);

    // Reverse: Fn = P.row(n - 1).asDiagonal() * Fn;
    bP.row(n - 1).noalias() += (Fn * bF.transpose()).diagonal();
    bF = P.row(n - 1).asDiagonal() * bF;

    // Reverse: update_f<is_solve>::apply(V.row(n - 1).transpose(), Y.row(n - 1), Z.row(n - 1), Fn);
    update_f<is_solve>::reverse(V.row(n - 1), Y.row(n - 1), Z.row(n - 1), bF, bV.row(n - 1), bY.row(n - 1), bZ.row(n - 1));
  }
}

template <bool is_solve, typename LowRank, typename RightHandSide, typename Work, typename RightHandSideInternal, typename LowRankOut,
          typename RightHandSideOut>
void backward_rev(const Eigen::MatrixBase<LowRank> &U,                    // (N, J)
                  const Eigen::MatrixBase<LowRank> &V,                    // (N, J)
                  const Eigen::MatrixBase<LowRank> &P,                    // (N-1, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,              // (N, Nrhs)
                  const Eigen::MatrixBase<RightHandSide> &Z,              // (N, Nrhs)
                  const Eigen::MatrixBase<Work> &F,                       // (N, J * Nrhs)
                  Eigen::MatrixBase<RightHandSideInternal> const &bZ_out, // (N, Nrhs)
                  Eigen::MatrixBase<LowRankOut> const &bU_out,            // (N, J)
                  Eigen::MatrixBase<LowRankOut> const &bV_out,            // (N, J)
                  Eigen::MatrixBase<LowRankOut> const &bP_out,            // (N-1, J)
                  Eigen::MatrixBase<RightHandSideOut> const &bY_out       // (N, Nrhs)  -  Must be the right shape already (and zeroed)
) {
  ASSERT_ROW_MAJOR(Work);

  typedef typename LowRank::Scalar Scalar;
  typedef typename Eigen::Matrix<Scalar, LowRank::ColsAtCompileTime, RightHandSide::ColsAtCompileTime> Inner;

  Eigen::Index N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST_MAT(LowRankOut, bU, N, J);
  CAST_MAT(LowRankOut, bV, N, J);
  CAST_MAT(LowRankOut, bP, N - 1, J);
  CAST_BASE(RightHandSideOut, bY);
  CAST_BASE(RightHandSideInternal, bZ);

  Inner Fn(J, nrhs), bF(J, nrhs);
  Eigen::Map<typename Eigen::internal::plain_row_type<Work>::type> ptr(Fn.data(), 1, J * nrhs);
  bF.setZero();
  for (Eigen::Index n = 0; n <= N - 2; ++n) {
    ptr = F.row(n);

    // Reverse: update_z<is_solve>::apply(V.row(n) * Fn, Z.row(n));
    update_z<is_solve>::apply(bZ.row(n) * (P.row(n).asDiagonal() * Fn).transpose(), bV.row(n));
    update_z<is_solve>::apply(V.row(n).transpose() * bZ.row(n), bF);

    // Reverse: Fn = P.row(n).asDiagonal() * Fn;
    bP.row(n).noalias() += (Fn * bF.transpose()).diagonal();
    bF = P.row(n).asDiagonal() * bF;

    // Reverse: update_f<is_solve>::apply(U.row(n + 1).transpose(), Y.row(n + 1), Z.row(n + 1), Fn);
    update_f<is_solve>::reverse(U.row(n + 1), Y.row(n + 1), Z.row(n + 1), bF, bU.row(n + 1), bY.row(n + 1), bZ.row(n + 1));
  }
}

} // namespace internal

} // namespace core
} // namespace celerite2

#endif // _CELERITE2_INTERNAL_HPP_DEFINED_
