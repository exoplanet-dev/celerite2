#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of matmul_upper", "[matmul_upper]", TestKernels) {
  SETUP_TEST(50);

  double max_resid;
  Matrix K, Z(Y.rows(), Y.cols()), F;
  to_dense(x, c, a, U, V, K);

  Matrix expect = K.triangularView<Eigen::StrictlyUpper>() * Y;

  SECTION("general") {
    Z.setZero();
    matmul_upper(x, c, U, V, Y, Z, F);
    max_resid = (expect - Z).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }

  SECTION("no grad") {
    Z.setZero();
    matmul_upper(x, c, U, V, Y, Z);
    max_resid = (expect - Z).array().abs().maxCoeff();
    REQUIRE(max_resid < 1e-12);
  }
}
