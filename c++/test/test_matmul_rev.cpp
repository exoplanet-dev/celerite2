#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of matmul_rev", "[matmul_rev]", TestKernels) {
  SETUP_TEST(10);

  CoeffVector bc(J);
  Vector ba(N), bx(N);
  LowRank bV(N, J), bU(N, J), bP(N - 1, J);
  Matrix Z, X, F, G, bZ(N, nrhs), bY(N, nrhs);

  // Required to compute the initial values
  matmul(x, c, a, U, V, Y, Z, X, F, G);

  auto func = [](auto x, auto c, auto a, auto U, auto V, auto Y, auto Z, auto X, auto F, auto G) {
    matmul(x, c, a, U, V, Y, Z, X, F, G);
    return std::make_tuple(Z);
  };

  auto rev = [](auto x, auto c, auto a, auto U, auto V, auto Y, auto Z, auto X, auto F, auto G, auto bZ, auto bx, auto bc, auto ba, auto bU, auto bV,
                auto bY) {
    matmul_rev(x, c, a, U, V, Y, Z, X, F, G, bZ, bx, bc, ba, bU, bV, bY);
    return std::make_tuple(bx, bc, ba, bU, bV, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(x, c, a, U, V, Y), std::make_tuple(Z, X, F, G), std::make_tuple(bZ),
                     std::make_tuple(bx, bc, ba, bU, bV, bY)));
}
