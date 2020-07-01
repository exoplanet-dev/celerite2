#ifndef _CELERITE2_INTERFACE_HPP_DEFINED_
#define _CELERITE2_INTERFACE_HPP_DEFINED_

#include <Eigen/Core>
#include "forward.hpp"

namespace celerite2 {
namespace core {

#define MakeEmptyWork                                                                                                                                \
  typedef typename LowRank::Scalar Scalar;                                                                                                           \
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Empty;                                                              \
  Empty

template <typename Diag, typename LowRank>
int factor(const Eigen::MatrixBase<Diag> &a,       // (N,)
           const Eigen::MatrixBase<LowRank> &U,    // (N, J)
           const Eigen::MatrixBase<LowRank> &V,    // (N, J)
           const Eigen::MatrixBase<LowRank> &P,    // (N-1, J)
           Eigen::MatrixBase<Diag> const &d_out,   // (N,)
           Eigen::MatrixBase<LowRank> const &W_out // (N, J)
) {
  MakeEmptyWork S;
  return factor<false>(a, U, V, P, d_out, W_out, S);
}

template <typename Diag, typename LowRank, typename RightHandSide>
void solve(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
           const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
           const Eigen::MatrixBase<Diag> &d,             // (N,)
           const Eigen::MatrixBase<LowRank> &W,          // (N, J)
           const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
           Eigen::MatrixBase<RightHandSide> const &X_out // (N, nrhs)
) {
  MakeEmptyWork F;
  solve<false>(U, P, d, W, Y, X_out, X_out, F, F);
}

template <typename Diag, typename LowRank, typename RightHandSide>
void dot_tril(const Eigen::MatrixBase<LowRank> &U,          // (N, J)
              const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
              const Eigen::MatrixBase<Diag> &d,             // (N,)
              const Eigen::MatrixBase<LowRank> &W,          // (N, J)
              const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
              Eigen::MatrixBase<RightHandSide> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork F;
  dot_tril<false>(U, P, d, W, Y, Z_out, F);
}

template <typename Diag, typename LowRank, typename RightHandSide>
void matmul(const Eigen::MatrixBase<Diag> &a,             // (N,)
            const Eigen::MatrixBase<LowRank> &U,          // (N, J)
            const Eigen::MatrixBase<LowRank> &V,          // (N, J)
            const Eigen::MatrixBase<LowRank> &P,          // (N-1, J)
            const Eigen::MatrixBase<RightHandSide> &Y,    // (N, nrhs)
            Eigen::MatrixBase<RightHandSide> const &X_out // (N, nrhs)
) {
  MakeEmptyWork F;
  matmul<false>(a, U, V, P, Y, X_out, X_out, F, F);
}

} // namespace core2
} // namespace celerite2

#endif // _CELERITE2_INTERFACE_HPP_DEFINED_