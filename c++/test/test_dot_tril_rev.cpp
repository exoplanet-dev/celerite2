#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of dot_tril_rev", "[dot_tril_rev]", TestKernels) {
  SETUP_TEST(10);

  Vector d, bd(N);
  LowRank W, bW(N, J), bU(N, J), bP(N - 1, J);
  Matrix S, Z, F, bZ(N, nrhs), bY(N, nrhs);

  // Required to compute the initial values
  celerite2::core2::factor(a, U, V, P, d, W, S);
  celerite2::core2::dot_tril(U, P, d, W, Y, Z, F);

  auto func = [](auto U, auto P, auto d, auto W, auto Y, auto Z, auto F) {
    celerite2::core2::dot_tril(U, P, d, W, Y, Z, F);
    return std::make_tuple(Z);
  };

  auto rev = [](auto U, auto P, auto d, auto W, auto Y, auto Z, auto F, auto bZ, auto bU, auto bP, auto bd, auto bW, auto bY) {
    celerite2::core2::dot_tril_rev(U, P, d, W, Y, Z, F, bZ, bU, bP, bd, bW, bY);
    return std::make_tuple(bU, bP, bd, bW, bY);
  };

  REQUIRE(check_grad(func, rev, std::make_tuple(U, P, d, W, Y), std::make_tuple(Z, F), std::make_tuple(bZ), std::make_tuple(bU, bP, bd, bW, bY)));
}
