#ifndef _CELERITE2_REVERSE_HPP_DEFINED_
#define _CELERITE2_REVERSE_HPP_DEFINED_

#include <Eigen/Core>
#include "internal.hpp"

namespace celerite2 {
namespace core2 {

template <typename Diag, typename LowRank, typename Work>
void factor_rev(const Eigen::MatrixBase<Diag> &a,         // (N,)
                const Eigen::MatrixBase<LowRank> &U,      // (N, J)
                const Eigen::MatrixBase<LowRank> &V,      // (N, J)
                const Eigen::MatrixBase<LowRank> &P,      // (N-1, J)
                const Eigen::MatrixBase<Diag> &d,         // (N,)
                const Eigen::MatrixBase<LowRank> &W,      // (N, J)
                const Eigen::MatrixBase<Work> &S,         // (N, J*J)
                const Eigen::MatrixBase<Diag> &bd,        // (N,)
                const Eigen::MatrixBase<LowRank> &bW,     // (N, J)
                Eigen::MatrixBase<Diag> const &ba_out,    // (N,)
                Eigen::MatrixBase<LowRank> const &bU_out, // (N, J)
                Eigen::MatrixBase<LowRank> const &bV_out, // (N, J)
                Eigen::MatrixBase<LowRank> const &bP_out  // (N-1, J)

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

template <typename Diag, typename LowRank, typename RightHandSide, typename Work>
void solve_rev(const Eigen::MatrixBase<LowRank> &U,           // (N, J)
               const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
               const Eigen::MatrixBase<Diag> &d,              // (N,)
               const Eigen::MatrixBase<LowRank> &W,           // (N, J)
               const Eigen::MatrixBase<RightHandSide> &Y,     // (N, nrhs)
               const Eigen::MatrixBase<RightHandSide> &X,     // (N, nrhs)
               const Eigen::MatrixBase<RightHandSide> &Z,     // (N, nrhs)
               const Eigen::MatrixBase<Work> &F,              // (N, J*nrhs)
               const Eigen::MatrixBase<Work> &G,              // (N, J*nrhs)
               const Eigen::MatrixBase<RightHandSide> &bX,    // (N, nrhs)
               Eigen::MatrixBase<LowRank> const &bU_out,      // (N, J)
               Eigen::MatrixBase<LowRank> const &bP_out,      // (N-1, J)
               Eigen::MatrixBase<Diag> const &bd_out,         // (N,)
               Eigen::MatrixBase<LowRank> const &bW_out,      // (N, J)
               Eigen::MatrixBase<RightHandSide> const &bY_out // (N, nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  int N = U.rows(), J = U.cols(), nrhs = Y.cols();
  CAST(LowRank, bU, N, J);
  CAST(LowRank, bP, N - 1, J);
  CAST(Diag, bd, N);
  CAST(LowRank, bW, N, J);
  CAST(RightHandSide, bY, N, nrhs);

  bU.setZero();
  bP.setZero();
  bd.setZero();
  bW.setZero();
  bY.setZero();

  RightHandSide bZ(N, nrhs);
  bZ.setZero();

  internal::backward_rev<true>(U, W, P, Z, X, G, bX, bU, bW, bP, bZ);

  // Reverse: X.array().colwise() /= d.array(); X = Z;
  bd.array() = -(X.array() * bX.array()).rowwise().sum();
  bZ.array() += bX.array().colwise() / d.array();

  internal::forward_rev<true>(U, W, P, Y, Z, F, bZ, bU, bW, bP, bY);
}

template <typename Diag, typename LowRank, typename RightHandSide, typename Work>
void dot_tril_rev(const Eigen::MatrixBase<LowRank> &U,           // (N, J)
                  const Eigen::MatrixBase<LowRank> &P,           // (N-1, J)
                  const Eigen::MatrixBase<Diag> &d,              // (N,)
                  const Eigen::MatrixBase<LowRank> &W,           // (N, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,     // (N, nrhs)
                  const Eigen::MatrixBase<RightHandSide> &Z,     // (N, nrhs)
                  const Eigen::MatrixBase<Work> &F,              // (N, J*nrhs)
                  const Eigen::MatrixBase<RightHandSide> &bZ,    // (N, nrhs)
                  Eigen::MatrixBase<LowRank> const &bU_out,      // (N, J)
                  Eigen::MatrixBase<LowRank> const &bP_out,      // (N-1, J)
                  Eigen::MatrixBase<Diag> const &bd_out,         // (N,)
                  Eigen::MatrixBase<LowRank> const &bW_out,      // (N, J)
                  Eigen::MatrixBase<RightHandSide> const &bY_out // (N, nrhs)
) {
  ASSERT_ROW_MAJOR(Work);

  CAST(RightHandSide, bY);
  CAST(Diag, bd, d.rows());

  Eigen::Matrix<typename Diag::Scalar, Diag::RowsAtCompileTime, 1> sqrtd = sqrt(d.array());

  // We need to repeat this calculation before running the backprop
  RightHandSide tmp = Y;
  tmp.array().colwise() *= sqrtd.array();

  // Run backprop
  bY = bZ;
  internal::forward_rev<false>(U, W, P, tmp, Z, F, bZ, bU_out, bW_out, bP_out, bY);

  // Update bY and bd based on tmp op above
  bd = 0.5 * (Y * bY.transpose()).diagonal().array() / sqrtd.array();
  bY.array().colwise() *= sqrtd.array();
}

} // namespace core2
} // namespace celerite2

#endif // _CELERITE2_REVERSE_HPP_DEFINED_
