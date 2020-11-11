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

/**
 * \brief Compute the Cholesky factorization of the system
 *
 * This computes `d` and `W` such that:
 *
 * `diag(a) + tril(U*V^T) + triu(V*U^T) = L*diag(d)*L^T`
 *
 * where
 *
 * `L = 1 + tril(U*W^T)`
 *
 * This can be safely applied in place: `d_out` can point to `a` and `W_out` can
 * point to `V`, and the memory will be reused.
 *
 * @param t     (N,): The input coordinates (must be sorted)
 * @param c     (J,): The transport coefficients
 * @param a     (N,): The diagonal component
 * @param U     (N, J): The first low rank matrix
 * @param V     (N, J): The second low rank matrix
 * @param d_out (N,): The diagonal component of the Cholesky factor
 * @param W_out (N, J): The second low rank component of the Cholesky factor
 */
template <typename Input, typename Coeffs, typename Diag, typename LowRank, typename DiagOut, typename LowRankOut>
Eigen::Index factor(const Eigen::MatrixBase<Input> &t,         // (N,)
                    const Eigen::MatrixBase<Coeffs> &c,        // (J,)
                    const Eigen::MatrixBase<Diag> &a,          // (N,)
                    const Eigen::MatrixBase<LowRank> &U,       // (N, J)
                    const Eigen::MatrixBase<LowRank> &V,       // (N, J)
                    Eigen::MatrixBase<DiagOut> const &d_out,   // (N,)
                    Eigen::MatrixBase<LowRankOut> const &W_out // (N, J)
) {
  MakeEmptyWork(Diag) S;
  return factor<false>(t, c, a, U, V, d_out, W_out, S);
}

/**
 * \brief Compute the solution of a lower triangular linear equation
 *
 * This computes `Z` such that:
 *
 * `Y = L * Y`
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
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void solve_lower(const Eigen::MatrixBase<Input> &t,               // (N,)
                 const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                 const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                 const Eigen::MatrixBase<LowRank> &W,             // (N, J)
                 const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
                 Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  solve_lower<false>(t, c, U, W, Y, Z_out, F);
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
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void solve_upper(const Eigen::MatrixBase<Input> &t,               // (N,)
                 const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                 const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                 const Eigen::MatrixBase<LowRank> &W,             // (N, J)
                 const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
                 Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  solve_upper<false>(t, c, U, W, Y, Z_out, F);
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
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void matmul_lower(const Eigen::MatrixBase<Input> &t,               // (N,)
                  const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                  const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                  const Eigen::MatrixBase<LowRank> &V,             // (N, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
                  Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  matmul_lower<false>(t, c, U, V, Y, Z_out, F);
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
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void matmul_upper(const Eigen::MatrixBase<Input> &t,               // (N,)
                  const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                  const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                  const Eigen::MatrixBase<LowRank> &V,             // (N, J)
                  const Eigen::MatrixBase<RightHandSide> &Y,       // (N, nrhs)
                  Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  matmul_upper<false>(t, c, U, V, Y, Z_out, F);
}

/**
 * \brief The general lower-triangular dot product of a rectangular celerite system
 *
 * @param t1     (N,): The left input coordinates (must be sorted)
 * @param t2     (M,): The right input coordinates (must be sorted)
 * @param c      (J,): The transport coefficients
 * @param U      (N, J): The first low rank matrix
 * @param V      (M, J): The second low rank matrix
 * @param Y      (M, Nrhs): The matrix that will be multiplied
 * @param Z_out  (N, Nrhs): The result of the operation
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void general_matmul_lower(const Eigen::MatrixBase<Input> &t1,              // (N,)
                          const Eigen::MatrixBase<Input> &t2,              // (M,)
                          const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                          const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                          const Eigen::MatrixBase<LowRank> &V,             // (M, J)
                          const Eigen::MatrixBase<RightHandSide> &Y,       // (M, nrhs)
                          Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  general_matmul_lower<false>(t1, t2, c, U, V, Y, Z_out, F);
}

/**
 * \brief The general upper-triangular dot product of a rectangular celerite system
 *
 * @param t1     (N,): The left input coordinates (must be sorted)
 * @param t2     (M,): The right input coordinates (must be sorted)
 * @param c      (J,): The transport coefficients
 * @param U      (N, J): The first low rank matrix
 * @param V      (M, J): The second low rank matrix
 * @param Y      (M, Nrhs): The matrix that will be multiplied
 * @param Z_out  (N, Nrhs): The result of the operation
 */
template <typename Input, typename Coeffs, typename LowRank, typename RightHandSide, typename RightHandSideOut>
void general_matmul_upper(const Eigen::MatrixBase<Input> &t1,              // (N,)
                          const Eigen::MatrixBase<Input> &t2,              // (M,)
                          const Eigen::MatrixBase<Coeffs> &c,              // (J,)
                          const Eigen::MatrixBase<LowRank> &U,             // (N, J)
                          const Eigen::MatrixBase<LowRank> &V,             // (M, J)
                          const Eigen::MatrixBase<RightHandSide> &Y,       // (M, nrhs)
                          Eigen::MatrixBase<RightHandSideOut> const &Z_out // (N, nrhs)
) {
  MakeEmptyWork(Input) F;
  general_matmul_upper<false>(t1, t2, c, U, V, Y, Z_out, F);
}

} // namespace core
} // namespace celerite2

#endif // _CELERITE2_INTERFACE_HPP_DEFINED_
