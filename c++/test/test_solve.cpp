#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <Eigen/Dense>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of solve", "[solve]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, S, F, G, X, Z;
  to_dense(a, U, V, P, K);

  // Do the solve using celerite
  int flag = factor(a, U, V, P, a, V, S);
  REQUIRE(flag == 0);

  // Brute force the solve
  Eigen::LDLT<Matrix> LDLT(K);
  Matrix expect = LDLT.solve(Y);

  SECTION("general") {
    solve(U, P, a, V, Y, X, Z, F, G);
    double resid = (expect - X).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }

  SECTION("no grad") {
    solve(U, P, a, V, Y, X);
    double resid = (expect - X).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }

  SECTION("inplace") {
    X = Y;
    solve(U, P, a, V, X, X);
    double resid = (expect - X).array().abs().maxCoeff();
    REQUIRE(resid < 1e-12);
  }
}
