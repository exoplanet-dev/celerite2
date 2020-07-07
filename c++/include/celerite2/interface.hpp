#ifndef _CELERITE2_INTERFACE_HPP_DEFINED_
#define _CELERITE2_INTERFACE_HPP_DEFINED_

#include <Eigen/Core>
#include "forward.hpp"

namespace celerite2 {
namespace core {

#define MakeEmptyWork(BaseType)                                                                                                                      \
  typedef typename BaseType::Scalar Scalar;                                                                                                          \
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Empty;                                                              \
  Empty

template <typename Diag, typename LowRank, typename DiagOut, typename LowRankOut>
Eigen::Index factor(const Eigen::MatrixBase<Diag> &a,          // (N,)
                    const Eigen::MatrixBase<LowRank> &U,       // (N, J)
                    const Eigen::MatrixBase<LowRank> &V,       // (N, J)
                    const Eigen::MatrixBase<LowRank> &P,       // (N-1, J)
                    Eigen::MatrixBase<DiagOut> const &d_out,   // (N,)
                    Eigen::MatrixBase<LowRankOut> const &W_out // (N, J)
) {
  MakeEmptyWork(Diag) S;
  return factor<false>(a, U, V, P, d_out, W_out, S);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void solve(const Eigen::MatrixBase<LowRank> &U,             // (N, J)
           const Eigen::MatrixBase<LowRank> &P,             // (N-1, J)
           const Eigen::MatrixBase<Diag> &d,                // (N,)
           const Eigen::MatrixBase<LowRank> &W,             // (N, J)
           const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
           Eigen::MatrixBase<RightHandSideOut> const &X_out // (N, nrhs)
) {
  MakeEmptyWork(Diag) F;
  solve<false>(U, P, d, W, Y, X_out, X_out, F, F);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename Norm, typename RightHandSideOut>
void norm(const Eigen::MatrixBase<LowRank> &U,             // (N, J)
          const Eigen::MatrixBase<LowRank> &P,             // (N-1, J)
          const Eigen::MatrixBase<Diag> &d,                // (N,)
          const Eigen::MatrixBase<LowRank> &W,             // (N, J)
          const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
          Eigen::MatrixBase<Norm> const &norm_out,         // (nrhs, nrhs)
          Eigen::MatrixBase<RightHandSideOut> const &X_out // (N, nrhs)
) {
  MakeEmptyWork(Diag) F;
  norm<false>(U, P, d, W, Y, norm_out, X_out, F);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void dot_tril(const Eigen::MatrixBase<LowRank> &U,             // (N, J)
              const Eigen::MatrixBase<LowRank> &P,             // (N-1, J)
              const Eigen::MatrixBase<Diag> &d,                // (N,)
              const Eigen::MatrixBase<LowRank> &W,             // (N, J)
              const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
              Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Diag) F;
  dot_tril<false>(U, P, d, W, Y, Z_out, F);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void matmul(const Eigen::MatrixBase<Diag> &a,                // (N,)
            const Eigen::MatrixBase<LowRank> &U,             // (N, J)
            const Eigen::MatrixBase<LowRank> &V,             // (N, J)
            const Eigen::MatrixBase<LowRank> &P,             // (N-1, J)
            const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
            Eigen::MatrixBase<RightHandSideOut> const &X_out // (N, nrhs)
) {
  MakeEmptyWork(Diag) F;
  matmul<false>(a, U, V, P, Y, X_out, X_out, F, F);
}

} // namespace core
} // namespace celerite2

#endif // _CELERITE2_INTERFACE_HPP_DEFINED_
