#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of general matmul", "[general_matmul]", TestKernels) {
  SETUP_TEST(50);

  double max_resid;
  Matrix K, Z1, Z2, F;
  to_dense(x, c, a - diag, U, V, K);

  general_lower_dot(x, x, c, U, V, Y, Z1, F);
  general_upper_dot(x, x, c, V, U, Y, Z2, F);

  SECTION("lower") {
    max_resid = (K.triangularView<Eigen::Lower>() * Y - Z1).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }

  SECTION("upper") {
    max_resid = (K.triangularView<Eigen::StrictlyUpper>() * Y - Z2).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }

  SECTION("full") {
    max_resid = (K * Y - Z1 - Z2).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }

  SECTION("no grad") {
    general_lower_dot(x, x, c, U, V, Y, Z1);
    general_upper_dot(x, x, c, V, U, Y, Z2);
    max_resid = (K * Y - Z1 - Z2).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }
}
