#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of matmul", "[matmul]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();

  typename decltype(kernel)::Vector a;
  typename decltype(kernel)::LowRank U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  Matrix K, Z;
  celerite2::core::to_dense(a, U, V, P, K);
  celerite2::core::matmul(a, U, V, P, Y, Z);

  double max_resid = (K * Y - Z).array().abs().maxCoeff();
  REQUIRE(max_resid < 1e-12);
}
