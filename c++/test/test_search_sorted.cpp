#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "helpers.hpp"
#include <celerite2/utils.hpp>

using namespace celerite2::test;

TEST_CASE("check the results of search_sorted", "[search_sorted]") {
  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  REQUIRE(celerite2::utils::search_sorted(x, x.minCoeff() - 1.0) == 0);
  REQUIRE(celerite2::utils::search_sorted(x, x.minCoeff()) == 1);
  REQUIRE(celerite2::utils::search_sorted(x, x.maxCoeff() + 1.0) == N);
  REQUIRE(celerite2::utils::search_sorted(x, x.maxCoeff()) == N);

  for (double v = x.minCoeff() + 1e-5; v <= x.maxCoeff() - 1e-5; v += 1e-3) {
    const int ind = celerite2::utils::search_sorted(x, v);
    REQUIRE(x(ind - 1) <= v);
    REQUIRE(x(ind) > v);
  }
}
