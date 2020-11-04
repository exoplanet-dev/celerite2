#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of dot_tril_rev", "[dot_tril_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc(J);
  Vector d, bd(N), bx(N);
  LowRank W, bW(N, J), bU(N, J);
  Matrix S, Z, F, bZ(N, nrhs), bY(N, nrhs);

  // Required to compute the initial values
  factor(x, c, a, U, V, d, W, S);
  dot_tril(x, c, U, d, W, Y, Z, F);

  auto func = [](auto x, auto c, auto U, auto d, auto W, auto Y, auto Z, auto F) {
    dot_tril(x, c, U, d, W, Y, Z, F);
    return std::make_tuple(Z);
  };

  auto rev = [](auto x, auto c, auto U, auto d, auto W, auto Y, auto Z, auto F, auto bZ, auto bx, auto bc, auto bU, auto bd, auto bW, auto bY) {
    dot_tril_rev(x, c, U, d, W, Y, Z, F, bZ, bx, bc, bU, bd, bW, bY);
    return std::make_tuple(bx, bc, bU, bd, bW, bY);
  };

  REQUIRE(
     check_grad(func, rev, std::make_tuple(x, c, U, d, W, Y), std::make_tuple(Z, F), std::make_tuple(bZ), std::make_tuple(bx, bc, bU, bd, bW, bY)));
}
