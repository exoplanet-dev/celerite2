#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <Eigen/Dense>
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of dot_tril", "[dot_tril]", TestKernels) {
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
  Eigen::LLT<Eigen::MatrixXd> LLT(K);
  Eigen::MatrixXd expect = LLT.matrixL() * Y;

  // Do the product using celerite
  Matrix Z(Y.rows(), Y.cols()), F(Y.rows(), Y.cols() * U.cols());
  celerite2::core2::dot_tril(U, P, a, V, Y, Z, F);
  double resid = (Z - expect).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);
}
