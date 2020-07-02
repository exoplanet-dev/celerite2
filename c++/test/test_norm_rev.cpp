#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <vector>
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of solve_rev", "[solve_rev]", TestKernels) {
  SETUP_TEST(10);

  Vector d, bd;
  LowRank W, bU, bP, bW;
  Matrix bY, bX(Y.cols(), Y.cols());
  Matrix S, F, G, X, Z;

  int flag = factor(a, U, V, P, d, W, S);
  REQUIRE(flag == 0);
  norm(U, P, d, W, Y, X, Z, F);

  auto func = [](auto U, auto P, auto d, auto W, auto Y, auto X, auto Z, auto F) {
    norm(U, P, d, W, Y, X, Z, F);
    return std::make_tuple(X);
  };

  auto rev = [](auto U, auto P, auto d, auto W, auto Y, auto X, auto Z, auto F, auto bX, auto bU, auto bP, auto bd, auto bW, auto bY) {
    norm_rev(U, P, d, W, Y, X, Z, F, bX, bU, bP, bd, bW, bY);
    return std::make_tuple(bU, bP, bd, bW, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(U, P, d, W, Y), std::make_tuple(X, Z, F), std::make_tuple(bX), std::make_tuple(bU, bP, bd, bW, bY)));
}