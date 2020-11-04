#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of matmul", "[matmul]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, Z, X, F, G;
  to_dense(x, c, a, U, V, K);

  SECTION("general") {
    matmul(x, c, a, U, V, Y, Z, X, F, G);
    double max_resid = (K * Y - Z).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }

  SECTION("no grad") {
    matmul(x, c, a, U, V, Y, Z);
    double max_resid = (K * Y - Z).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }
}
