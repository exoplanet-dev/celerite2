#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of matmul_lower_rev", "[matmul_lower_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc(J);
  Vector bx(N);
  LowRank bV(N, J), bU(N, J);
  Matrix Z(N, nrhs), F, bZ(N, nrhs), bY(N, nrhs);

  // Required to compute the initial values
  matmul_lower(x, c, U, V, Y, Z, F);

  auto func = [](auto x, auto c, auto U, auto V, auto Y, auto Z, auto F) {
    Z.setZero();
    matmul_lower(x, c, U, V, Y, Z, F);
    return std::make_tuple(Z);
  };

  auto rev = [](auto x, auto c, auto U, auto V, auto Y, auto Z, auto F, auto bZ, auto bx, auto bc, auto bU, auto bV, auto bY) {
    matmul_lower_rev(x, c, U, V, Y, Z, F, bZ, bx, bc, bU, bV, bY);
    return std::make_tuple(bx, bc, bU, bV, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(x, c, U, V, Y), std::make_tuple(Z, F), std::make_tuple(bZ), std::make_tuple(bx, bc, bU, bV, bY)));
}
