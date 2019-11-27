#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of factor_grad", "[factor_grad]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Eigen::VectorXd x, diag;
  Eigen::MatrixXd Y;
  std::tie(x, diag, Y) = get_data(10);
  const int N          = x.rows();

  Eigen::VectorXd a;
  Eigen::MatrixXd U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);
  const int J          = U.cols();

  Eigen::VectorXd d = a, d0;
  Eigen::MatrixXd S, W = V, U0, P0, W0;
  int flag = celerite::core::factor(U, P, d, W, S);
  REQUIRE(flag == 0);

  Eigen::VectorXd ba(N);
  Eigen::MatrixXd bV(N, J), bU, bP;

  // Compute numerical derivatives
  const double eps = 1.234e-8;
  const double tol = 500 * eps;
  std::vector<Eigen::MatrixXd> ddda(N), dWda(N);
  std::vector<std::vector<Eigen::MatrixXd>> dddV(N), dWdV(N), dddU(N), dWdU(N), dddP(N - 1), dWdP(N - 1);
  for (int n = 0; n < N; ++n) {
    d0 = a;
    W0 = V;
    U0 = U;
    P0 = P;
    d0(n) += eps;
    celerite::core::factor(U0, P0, d0, W0, S);
    ddda[n] = (d0 - d) / eps;
    dWda[n] = (W0 - W) / eps;

    dddV[n].resize(J);
    dWdV[n].resize(J);
    dddU[n].resize(J);
    dWdU[n].resize(J);
    for (int j = 0; j < J; ++j) {
      d0 = a;
      W0 = V;
      W0(n, j) += eps;
      celerite::core::factor(U0, P0, d0, W0, S);
      dddV[n][j] = (d0 - d) / eps;
      dWdV[n][j] = (W0 - W) / eps;

      // U
      d0 = a;
      W0 = V;
      U0(n, j) += eps;
      celerite::core::factor(U0, P0, d0, W0, S);
      U0(n, j) -= eps;
      dddU[n][j] = (d0 - d) / eps;
      dWdU[n][j] = (W0 - W) / eps;
    }

    // P
    if (n < N - 1) {
      dddP[n].resize(J);
      dWdP[n].resize(J);
      for (int j = 0; j < J; ++j) {
        d0 = a;
        W0 = V;
        P0(n, j) += eps;
        celerite::core::factor(U0, P0, d0, W0, S);
        P0(n, j) -= eps;
        dddP[n][j] = (d0 - d) / eps;
        dWdP[n][j] = (W0 - W) / eps;
      }
    }
  }

  // Test these against the backpropagated derivatives
  for (int n = 0; n < N; ++n) {
    // a
    ba.setZero();
    bV.setZero();
    ba(n) = 1.0;
    celerite::core::factor_grad(U, P, d, W, S, bU, bP, ba, bV);
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
      ba.setZero();
      bV.setZero();
      bV(n, k) = 1.0;
      celerite::core::factor_grad(U, P, d, W, S, bU, bP, ba, bV);
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
