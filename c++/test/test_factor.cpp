#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <Eigen/Dense>
#include <celerite2/core2.hpp>
using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of factor", "[factor]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, S;
  celerite2::core2::to_dense(a, U, V, P, K);

  // Do the Cholesky using celerite
  int flag = celerite2::core2::factor(a, U, V, P, a, V, S);
  REQUIRE(flag == 0);

  // Reconstruct the L matrix
  Matrix UWT;
  celerite2::core2::to_dense(Eigen::VectorXd::Ones(N), U, V, P, UWT);
  UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Matrix> LDLT(K);
  Eigen::MatrixXd matrixL = LDLT.matrixL();

  // Check that the lower triangle is correct
  double resid = (matrixL - UWT).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);

  // Check that the diagonal is correct
  double diag_resid = (LDLT.vectorD() - a).array().abs().maxCoeff();
  REQUIRE(diag_resid < 1e-12);
}
