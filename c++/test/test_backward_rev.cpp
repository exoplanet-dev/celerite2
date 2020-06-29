#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/core2.hpp>

using namespace celerite2::test;

TEMPLATE_LIST_TEST_CASE("check the results of backward_rev", "[backward_rev]", TestKernels) {
#define BUILD_TEST(is_solve)                                                                                                                         \
  {                                                                                                                                                  \
    SETUP_TEST(10);                                                                                                                                  \
                                                                                                                                                     \
    Matrix Z(N, nrhs), F, bZ(N, nrhs), bY(N, nrhs);                                                                                                  \
    LowRank bU(N, J), bV(N, J), bP(N - 1, J);                                                                                                        \
                                                                                                                                                     \
    Z.setZero();                                                                                                                                     \
    celerite2::core2::internal::backward<is_solve>(U, V, P, Y, Z, F);                                                                                \
                                                                                                                                                     \
    auto func = [](auto U, auto V, auto P, auto Y, auto Z, auto F) {                                                                                 \
      celerite2::core2::internal::backward<is_solve>(U, V, P, Y, Z, F);                                                                              \
      return std::make_tuple(Z);                                                                                                                     \
    };                                                                                                                                               \
                                                                                                                                                     \
    auto rev = [](auto U, auto V, auto P, auto Y, auto Z, auto F, auto bZ, auto bU, auto bV, auto bP, auto bY) {                                     \
      celerite2::core2::internal::backward_rev<is_solve>(U, V, P, Y, Z, F, bZ, bU, bV, bP, bY);                                                      \
      return std::make_tuple(bU, bV, bP, bY);                                                                                                        \
    };                                                                                                                                               \
                                                                                                                                                     \
    REQUIRE(check_grad(func, rev, std::make_tuple(U, V, P, Y), std::make_tuple(Z, F), std::make_tuple(bZ), std::make_tuple(bU, bV, bP, bY)));        \
  }

  SECTION("solve: ") { BUILD_TEST(true); }
  SECTION("not solve: ") { BUILD_TEST(false); }
}
