#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"
#include <Eigen/Dense>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of factor", "[factor]", TestKernels) {
  SETUP_TEST(50);

  Vector d;
  LowRank W;
  Matrix K, S;
  to_dense(x, c, a, U, V, K);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Matrix> LDLT(K);
  Eigen::MatrixXd matrixL = LDLT.matrixL();

  SECTION("general") {
    // Do the Cholesky using celerite
    int flag = factor(x, c, a, U, V, d, W, S);
    REQUIRE(flag == 0);

    // Reconstruct the L matrix
    Matrix UWT;
    to_dense(x, c, Eigen::VectorXd::Ones(N), U, W, UWT);
    UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

    // Check that the lower triangle is correct
    double resid = (matrixL - UWT).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);

    // Check that the diagonal is correct
    double diag_resid = (LDLT.vectorD() - d).array().abs().maxCoeff();
    REQUIRE(diag_resid < 1e-12);
  }

  SECTION("no grad") {
    // Do the Cholesky using celerite
    int flag = factor(x, c, a, U, V, d, W);
    REQUIRE(flag == 0);

    // Reconstruct the L matrix
    Matrix UWT;
    to_dense(x, c, Eigen::VectorXd::Ones(N), U, W, UWT);
    UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

    // Check that the lower triangle is correct
    double resid = (matrixL - UWT).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);

    // Check that the diagonal is correct
    double diag_resid = (LDLT.vectorD() - d).array().abs().maxCoeff();
    REQUIRE(diag_resid < 1e-12);
  }

  SECTION("inplace") {
    // Do the Cholesky using celerite
    int flag = factor(x, c, a, U, V, a, V, S);
    REQUIRE(flag == 0);

    // Reconstruct the L matrix
    Matrix UWT;
    to_dense(x, c, Eigen::VectorXd::Ones(N), U, V, UWT);
    UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

    // Check that the lower triangle is correct
    double resid = (matrixL - UWT).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);

    // Check that the diagonal is correct
    double diag_resid = (LDLT.vectorD() - a).array().abs().maxCoeff();
    REQUIRE(diag_resid < 1e-12);
  }
}
