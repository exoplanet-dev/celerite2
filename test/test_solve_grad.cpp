#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of solve_grad", "[solve_grad]", TestKernels) {
  auto kernel = TestType::get_kernel();

  typedef typename decltype(kernel)::LowRank LowRank;

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data(10);
  const int N          = x.rows();
  const int nrhs       = Y.cols();

  Vector a;
  LowRank U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);
  const int J          = U.cols();

  Vector d  = a, d0;
  LowRank W = V, U0, P0, W0;
  Matrix S;
  int flag = celerite::core::factor(U, P, d, W, S);
  REQUIRE(flag == 0);

  Matrix Z = Y, F, G, Z0, F0, G0;
  celerite::core::solve(U, P, d, W, Z, F, G);

  Vector bd;
  LowRank bU, bP, bW;
  Matrix bZ(N, nrhs), bY;

  // Compute numerical derivatives
  const double eps = 1.234e-8;
  const double tol = 500 * eps;

  std::vector<Matrix> dZdd(N);
  std::vector<std::vector<Matrix>> dZdU(N), dZdP(N - 1), dZdW(N), dZdY(N);

  d0 = d;
  W0 = W;
  U0 = U;
  P0 = P;

  for (int n = 0; n < N; ++n) {
    Z0 = Y;
    d0(n) += eps;
    celerite::core::solve(U0, P0, d0, W0, Z0, F0, G0);
    d0(n) -= eps;
    dZdd[n] = (Z0 - Z) / eps;

    dZdU[n].resize(J);
    dZdW[n].resize(J);
    for (int j = 0; j < J; ++j) {
      Z0 = Y;
      U0(n, j) += eps;
      celerite::core::solve(U0, P0, d0, W0, Z0, F0, G0);
      U0(n, j) -= eps;
      dZdU[n][j] = (Z0 - Z) / eps;

      Z0 = Y;
      W0(n, j) += eps;
      celerite::core::solve(U0, P0, d0, W0, Z0, F0, G0);
      W0(n, j) -= eps;
      dZdW[n][j] = (Z0 - Z) / eps;
    }

    // P
    if (n < N - 1) {
      dZdP[n].resize(J);
      for (int j = 0; j < J; ++j) {
        Z0 = Y;
        P0(n, j) += eps;
        celerite::core::solve(U0, P0, d0, W0, Z0, F0, G0);
        P0(n, j) -= eps;
        dZdP[n][j] = (Z0 - Z) / eps;
      }
    }

    dZdY[n].resize(nrhs);
    for (int j = 0; j < nrhs; ++j) {
      Z0 = Y;
      Z0(n, j) += eps;
      celerite::core::solve(U0, P0, d0, W0, Z0, F0, G0);
      Z0(n, j) -= eps;
      dZdY[n][j] = (Z0 - Z) / eps;
    }
  }

  // Test these against the backpropagated derivatives
  for (int n = 0; n < N; ++n) {
    for (int k = 0; k < nrhs; ++k) {
      bZ.setZero();
      bZ(n, k) = 1.0;
      celerite::core::solve_grad(U, P, d, W, Z, F, G, bZ, bU, bP, bd, bW, bY);
      for (int m = 0; m < N; ++m) {
        REQUIRE(std::abs(dZdd[m](n, k) - bd(m)) < tol);
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
