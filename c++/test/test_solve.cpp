#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <Eigen/Dense>
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of solve", "[solve]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, S, F, G, X, Z;
  celerite2::core2::to_dense(a, U, V, P, K);

  // Do the solve using celerite
  int flag = celerite2::core2::factor(a, U, V, P, a, V, S);
  REQUIRE(flag == 0);
  celerite2::core2::solve(U, P, a, V, Y, X, Z, F, G);

  // Brute force the solve
  Eigen::LDLT<Matrix> LDLT(K);
  Matrix expect = LDLT.solve(Y);

  // Check the result
  double resid = (expect - X).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);
}
