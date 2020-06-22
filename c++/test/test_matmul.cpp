#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <celerite2/core.hpp>
#include <celerite2/core2.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of matmul", "[matmul]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();

  typename decltype(kernel)::Vector a;
  typename decltype(kernel)::LowRank U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  Matrix K, Z(Y.rows(), Y.cols()), X(Y.rows(), Y.cols()), F(U.rows(), U.cols() * Y.cols()), G(U.rows(), U.cols() * Y.cols());
  celerite2::core::to_dense(a, U, V, P, K);
  celerite2::core2::matmul(a, U, V, P, Y, Z, X, F, G);

  double max_resid = (K * Y - Z).array().abs().maxCoeff();
  REQUIRE(max_resid < 1e-12);
}
