#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of solve_rev", "[solve_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc;
  Vector d, bd, bx;
  LowRank W, bU, bW;
  Matrix bY, bX(Y.rows(), Y.cols());
  Matrix S, F, G, X, Z;

  int flag = factor(x, c, a, U, V, d, W, S);
  REQUIRE(flag == 0);
  solve(x, c, U, d, W, Y, X, Z, F, G);

  auto func = [](auto x, auto c, auto U, auto d, auto W, auto Y, auto X, auto Z, auto F, auto G) {
    solve(x, c, U, d, W, Y, X, Z, F, G);
    return std::make_tuple(X);
  };

  auto rev = [](auto x, auto c, auto U, auto d, auto W, auto Y, auto X, auto Z, auto F, auto G, auto bX, auto bx, auto bc, auto bU, auto bd, auto bW,
                auto bY) {
    solve_rev(x, c, U, d, W, Y, X, Z, F, G, bX, bx, bc, bU, bd, bW, bY);
    return std::make_tuple(bx, bc, bU, bd, bW, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(x, c, U, d, W, Y), std::make_tuple(X, Z, F, G), std::make_tuple(bX),
                     std::make_tuple(bx, bc, bU, bd, bW, bY)));
}
