#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/core2.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of factor_grad", "[factor_grad]", TestKernels) {
  auto kernel = TestType::get_kernel();

  typedef typename decltype(kernel)::LowRank LowRank;

  const double eps = 1.234e-8;
  const double tol = 500 * eps;

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data(10);
  const int N          = x.rows();

  Vector a;
  LowRank U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);
  const int J          = U.cols();

  Vector d, d0;
  LowRank W, W0;
  Matrix S, S0;

  // Make sure everything is the right size
  d.resize(N);
  W.resize(N, J);
  S.resize(N, J * J);
  d0.resize(N);
  W0.resize(N, J);
  S0.resize(N, J * J);

  // Compute the reference matrices
  int flag = celerite2::core2::factor(a, U, V, P, d0, W0, S0);
  REQUIRE(flag == 0);

  // Compute numerical derivatives
  std::vector<Vector> ddda(N);
  std::vector<LowRank> dWda(N);
  std::vector<std::vector<Vector>> dddV(N), dddU(N), dddP(N - 1);
  std::vector<std::vector<LowRank>> dWdV(N), dWdU(N), dWdP(N - 1);
  for (int n = 0; n < N; ++n) {
    a(n) += eps;
    celerite2::core2::factor(a, U, V, P, d, W, S);
    a(n) -= eps;
    ddda[n] = (d - d0) / eps;
    dWda[n] = (W - W0) / eps;

    dddV[n].resize(J);
    dWdV[n].resize(J);
    dddU[n].resize(J);
    dWdU[n].resize(J);
    for (int j = 0; j < J; ++j) {
      // U
      U(n, j) += eps;
      celerite2::core2::factor(a, U, V, P, d, W, S);
      U(n, j) -= eps;
      dddU[n][j] = (d - d0) / eps;
      dWdU[n][j] = (W - W0) / eps;

      // V
      V(n, j) += eps;
      celerite2::core2::factor(a, U, V, P, d, W, S);
      V(n, j) -= eps;
      dddV[n][j] = (d - d0) / eps;
      dWdV[n][j] = (W - W0) / eps;
    }

    // P
    if (n < N - 1) {
      dddP[n].resize(J);
      dWdP[n].resize(J);
      for (int j = 0; j < J; ++j) {
        P(n, j) += eps;
        celerite2::core2::factor(a, U, V, P, d, W, S);
        P(n, j) -= eps;
        dddP[n][j] = (d - d0) / eps;
        dWdP[n][j] = (W - W0) / eps;
      }
    }
  }

  Vector ba(N), bd(N);
  LowRank bU(N, J), bV(N, J), bP(N - 1, J), bW(N, J);

  // Test these against the backpropagated derivatives
  for (int n = 0; n < N; ++n) {
    // a
    bd.setZero();
    bW.setZero();
    bd(n) = 1.0;
    celerite2::core2::factor_rev(a, U, V, P, d0, W0, S0, bd, bW, ba, bU, bV, bP);
    for (int m = 0; m < N; ++m) {
      REQUIRE(std::abs(ddda[m](n) - ba(m)) < tol);
      for (int j = 0; j < J; ++j) {
        REQUIRE(std::abs(dddV[m][j](n) - bV(m, j)) < tol);
        REQUIRE(std::abs(dddU[m][j](n) - bU(m, j)) < tol);
      }
      if (m < N - 1) {
        for (int j = 0; j < J; ++j) { REQUIRE(std::abs(dddP[m][j](n) - bP(m, j)) < tol); }
      }
    }

    // W
    for (int k = 0; k < J; ++k) {
      bd.setZero();
      bW.setZero();
      bW(n, k) = 1.0;
      celerite2::core2::factor_rev(a, U, V, P, d0, W0, S0, bd, bW, ba, bU, bV, bP);
      for (int m = 0; m < N; ++m) {
        REQUIRE(std::abs(dWda[m](n, k) - ba(m)) < tol);
        for (int j = 0; j < J; ++j) {
          REQUIRE(std::abs(dWdV[m][j](n, k) - bV(m, j)) < tol);
          REQUIRE(std::abs(dWdU[m][j](n, k) - bU(m, j)) < tol);
        }
        if (m < N - 1) {
          for (int j = 0; j < J; ++j) { REQUIRE(std::abs(dWdP[m][j](n, k) - bP(m, j)) < tol); }
        }
      }
    }
  }
}
