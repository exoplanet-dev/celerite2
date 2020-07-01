#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of matmul_rev", "[matmul_rev]", TestKernels) {
  SETUP_TEST(10);

  Vector ba(N);
  LowRank bV(N, J), bU(N, J), bP(N - 1, J);
  Matrix Z, X, F, G, bZ(N, nrhs), bY(N, nrhs);

  // Required to compute the initial values
  matmul(a, U, V, P, Y, Z, X, F, G);

  auto func = [](auto a, auto U, auto V, auto P, auto Y, auto Z, auto X, auto F, auto G) {
    matmul(a, U, V, P, Y, Z, X, F, G);
    return std::make_tuple(Z);
  };

  auto rev = [](auto a, auto U, auto V, auto P, auto Y, auto Z, auto X, auto F, auto G, auto bZ, auto ba, auto bU, auto bV, auto bP, auto bY) {
    matmul_rev(a, U, V, P, Y, Z, X, F, G, bZ, ba, bU, bV, bP, bY);
    return std::make_tuple(ba, bU, bV, bP, bY);
  };

  REQUIRE(
     check_grad(func, rev, std::make_tuple(a, U, V, P, Y), std::make_tuple(Z, X, F, G), std::make_tuple(bZ), std::make_tuple(ba, bU, bV, bP, bY)));
}
