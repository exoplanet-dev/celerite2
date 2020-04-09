#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <Eigen/Dense>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of factor", "[factor]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  typename decltype(kernel)::Vector a, d;
  typename decltype(kernel)::LowRank U, V, P, W;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  Matrix K, S;
  celerite2::core::to_dense(a, U, V, P, K);

  // Check the no-grad version first
  W        = V;
  d        = a;
  int flag = celerite2::core::factor(U, P, d, W);
  REQUIRE(flag == 0);

  // Do the Cholesky using celerite
  flag = celerite2::core::factor(U, P, a, V, S);
  REQUIRE(flag == 0);

  // Make sure that the no-grad version gives the right answer
  REQUIRE((a - d).array().abs().maxCoeff() < 1e-12);
  REQUIRE((V - W).array().abs().maxCoeff() < 1e-12);

  // Reconstruct the L matrix
  Matrix UWT;
  celerite2::core::to_dense(Eigen::VectorXd::Ones(N), U, V, P, UWT);
  UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Matrix> LDLT(K);
  Eigen::MatrixXd matrixL = LDLT.matrixL();

  // Check that the lower triangle is correct
  double resid = (matrixL - UWT).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);

  // Check that the diagonal is correct
  double diag_resid = (LDLT.vectorD() - a).array().abs().maxCoeff();
  REQUIRE(diag_resid < 1e-12);
}
