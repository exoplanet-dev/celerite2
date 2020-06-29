#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of factor_rev", "[factor_rev]", TestKernels) {
  SETUP_TEST(10);

  Vector d, bd(N), ba(N);
  LowRank W, bW(N, J), bU(N, J), bV(N, J), bP(N - 1, J);
  Matrix S;

  auto func = [](auto a, auto U, auto V, auto P, auto d, auto W, auto S) {
    celerite2::core2::factor(a, U, V, P, d, W, S);
    return std::make_tuple(d, W);
  };

  auto rev = [](auto a, auto U, auto V, auto P, auto d, auto W, auto S, auto bd, auto bW, auto ba, auto bU, auto bV, auto bP) {
    celerite2::core2::factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP);
    return std::make_tuple(ba, bU, bV, bP);
  };

  // Required to compute the initial values
  int flag = celerite2::core2::factor(a, U, V, P, d, W, S);
  REQUIRE(flag == 0);

  REQUIRE(check_grad(func, rev, std::make_tuple(a, U, V, P), std::make_tuple(d, W, S), std::make_tuple(bd, bW), std::make_tuple(ba, bU, bV, bP)));
}
