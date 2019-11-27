#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <Eigen/Dense>
#include <celerite2/terms.hpp>
#include <celerite2/core.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of factor", "[factor]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Eigen::VectorXd x, diag;
  Eigen::MatrixXd Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  Eigen::VectorXd a;
  Eigen::MatrixXd U, V, P;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  Eigen::MatrixXd K, S;
  celerite::core::to_dense(a, U, V, P, K);

  // Do the Cholesky using celerite
  int flag = celerite::core::factor(U, P, a, V, S);
  REQUIRE(flag == 0);

  // Reconstruct the L matrix
  Eigen::MatrixXd UWT;
  celerite::core::to_dense(Eigen::VectorXd::Ones(N), U, V, P, UWT);
  UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Eigen::MatrixXd> LDLT(K);
  Eigen::MatrixXd matrixL = LDLT.matrixL();

  // Check that the lower triangle is correct
  auto resid = (matrixL - UWT).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);

  // Check that the diagonal is correct
  auto diag_resid = (LDLT.vectorD() - a).array().abs().maxCoeff();
  REQUIRE(diag_resid < 1e-12);
}
