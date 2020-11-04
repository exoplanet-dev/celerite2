#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of factor_rev", "[factor_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc(J);
  Vector d, bd(N), bx(N), ba(N);
  LowRank W, bW(N, J), bU(N, J), bV(N, J);
  Matrix S;

  auto func = [](auto x, auto c, auto a, auto U, auto V, auto d, auto W, auto S) {
    factor(x, c, a, U, V, d, W, S);
    return std::make_tuple(d, W);
  };

  auto rev = [](auto x, auto c, auto a, auto U, auto V, auto d, auto W, auto S, auto bd, auto bW, auto bx, auto bc, auto ba, auto bU, auto bV) {
    factor_rev(x, c, a, U, V, d, W, S, bd, bW, bx, bc, ba, bU, bV);
    return std::make_tuple(bx, bc, ba, bU, bV);
  };

  // Required to compute the initial values
  int flag = factor(x, c, a, U, V, d, W, S);
  REQUIRE(flag == 0);

  REQUIRE(
     check_grad(func, rev, std::make_tuple(x, c, a, U, V), std::make_tuple(d, W, S), std::make_tuple(bd, bW), std::make_tuple(bx, bc, ba, bU, bV)));
}
