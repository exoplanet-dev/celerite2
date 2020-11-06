#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <Eigen/Dense>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of solve_upper", "[solve_upper]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, S, Z, F;
  to_dense(x, c, a, U, V, K);

  // Do the Cholesky using celerite
  int flag = factor(x, c, a, U, V, a, V, S);
  REQUIRE(flag == 0);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Eigen::MatrixXd> LDLT(K);
  Eigen::MatrixXd expect = LDLT.matrixL().transpose().solve(Y);

  SECTION("general") {
    solve_upper(x, c, U, V, Y, Z, F);
    double resid = (Z - expect).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }

  SECTION("no grad") {
    solve_upper(x, c, U, V, Y, Z);
    double resid = (Z - expect).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }

  SECTION("inplace") {
    Z = Y;
    solve_upper(x, c, U, V, Z, Z, F);
    double resid = (Z - expect).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }
}
