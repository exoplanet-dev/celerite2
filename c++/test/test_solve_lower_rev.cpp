#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of solve_lower_rev", "[solve_lower_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc;
  Vector d, bd, bx;
  LowRank W, bU, bW;
  Matrix bY, bZ(Y.rows(), Y.cols());
  Matrix S, F, G, Z;

  int flag = factor(x, c, a, U, V, d, W, S);
  REQUIRE(flag == 0);
  solve_lower(x, c, U, W, Y, Z, F);

  auto func = [](auto x, auto c, auto U, auto W, auto Y, auto Z, auto F) {
    solve_lower(x, c, U, W, Y, Z, F);
    return std::make_tuple(Z);
  };

  auto rev = [](auto x, auto c, auto U, auto W, auto Y, auto Z, auto F, auto bZ, auto bx, auto bc, auto bU, auto bW, auto bY) {
    solve_lower_rev(x, c, U, W, Y, Z, F, bZ, bx, bc, bU, bW, bY);
    return std::make_tuple(bx, bc, bU, bW, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(x, c, U, W, Y), std::make_tuple(Z, F), std::make_tuple(bZ), std::make_tuple(bx, bc, bU, bW, bY)));
}
