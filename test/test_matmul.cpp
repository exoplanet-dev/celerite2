#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <celerite2/terms.hpp>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of matmul", "[matmul]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Eigen::VectorXd x, diag;
  Eigen::MatrixXd Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  Eigen::VectorXd a;
  Eigen::MatrixXd U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  Eigen::MatrixXd K, Z;
  celerite::core::to_dense(a, U, V, P, K);
  celerite::core::matmul(a, U, V, P, Y, Z);

  double max_resid = (K * Y - Z).array().abs().maxCoeff();
  REQUIRE(max_resid < 1e-12);
}
