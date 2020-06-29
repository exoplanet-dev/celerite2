#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of matmul", "[matmul]", TestKernels) {
  SETUP_TEST(50);

  Matrix K, Z, X, F, G;
  celerite2::core2::to_dense(a, U, V, P, K);
  celerite2::core2::matmul(a, U, V, P, Y, Z, X, F, G);

  double max_resid = (K * Y - Z).array().abs().maxCoeff();
  REQUIRE(max_resid < 1e-12);
}
