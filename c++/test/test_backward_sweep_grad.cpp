#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of backward_sweep_grad", "[backward_sweep_grad]", TestKernels) {
  auto kernel = TestType::get_kernel();

  typedef typename decltype(kernel)::LowRank LowRank;

  constexpr bool is_solve = false;

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();
  const int nrhs       = Y.cols();

  Vector a;
  LowRank U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);
  const int J          = U.cols();

  Vector d  = a;
  LowRank W = V, U0, P0, W0;
  Matrix S;
  int flag = celerite2::core::factor(U, P, d, W, S);
  REQUIRE(flag == 0);

  Matrix Z = Y, F, Y0, Z0, F0;
  celerite2::core::internal::backward_sweep<is_solve, true>(U, P, W, Y, Z, F);

  LowRank bU(N, J), bP(N - 1, J), bW(N, J);
  Matrix bZ(N, nrhs), bY(N, nrhs);

  // Compute numerical derivatives
  const double eps = 1.234e-8;
  const double tol = 500 * eps;

  std::vector<std::vector<Matrix>> dZdU(N), dZdP(N - 1), dZdW(N), dZdY(N);

  W0 = W;
  U0 = U;
  P0 = P;
  Y0 = Y;

  for (int n = 0; n < N; ++n) {
    Z0 = Y;

    dZdU[n].resize(J);
    dZdW[n].resize(J);
    for (int j = 0; j < J; ++j) {
      Z0 = Y;
      U0(n, j) += eps;
      celerite2::core::internal::backward_sweep<is_solve, true>(U0, P0, W0, Y0, Z0, F0);
      U0(n, j) -= eps;
      dZdU[n][j] = (Z0 - Z) / eps;

      Z0 = Y;
      W0(n, j) += eps;
      celerite2::core::internal::backward_sweep<is_solve, true>(U0, P0, W0, Y0, Z0, F0);
      W0(n, j) -= eps;
      dZdW[n][j] = (Z0 - Z) / eps;
    }

    // P
    if (n < N - 1) {
      dZdP[n].resize(J);
      for (int j = 0; j < J; ++j) {
        Z0 = Y;
        P0(n, j) += eps;
        celerite2::core::internal::backward_sweep<is_solve, true>(U0, P0, W0, Y0, Z0, F0);
        P0(n, j) -= eps;
        dZdP[n][j] = (Z0 - Z) / eps;
      }
    }

    dZdY[n].resize(nrhs);
    for (int j = 0; j < nrhs; ++j) {
      Z0 = Y;
      Z0(n, j) += eps;
      Y0(n, j) += eps;
      celerite2::core::internal::backward_sweep<is_solve, true>(U0, P0, W0, Y0, Z0, F0);
      Z0(n, j) -= eps;
      Y0(n, j) -= eps;
      dZdY[n][j] = (Z0 - Z) / eps;
    }
  }

  // Test these against the backpropagated derivatives
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < nrhs; ++k) {
      bZ.setZero();
      bZ(n, k) = 1.0;
      bU.setZero();
      bP.setZero();
      bW.setZero();
      bY.setZero();
      celerite2::core::internal::backward_sweep_grad<is_solve>(U, P, W, Y, Z, F, bZ, bU, bP, bW, bY);
      for (int m = 0; m < N; ++m) {
        for (int j = 0; j < J; ++j) {
          REQUIRE(std::abs(dZdU[m][j](n, k) - bU(m, j)) < tol);
          REQUIRE(std::abs(dZdW[m][j](n, k) - bW(m, j)) < tol);
        }
        if (m < N - 1) {
          for (int j = 0; j < J; ++j) { REQUIRE(std::abs(dZdP[m][j](n, k) - bP(m, j)) < tol); }
        }
        for (int j = 0; j < nrhs; ++j) { REQUIRE(std::abs(dZdY[m][j](n, k) - bY(m, j)) < tol); }
      }
    }
  }
}
