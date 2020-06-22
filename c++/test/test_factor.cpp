#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <Eigen/Dense>
#include <celerite2/core2.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of factor", "[factor]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  typename decltype(kernel)::Vector a, d;
  typename decltype(kernel)::LowRank U, V, P, W;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);
  const int J          = U.cols();

  Matrix K, S;
  celerite2::core2::to_dense(a, U, V, P, K);

  // Make sure everything is the right size
  d.resize(N);
  W.resize(N, J);
  S.resize(N, J * J);

  // Do the Cholesky using celerite
  int flag = celerite2::core2::factor(a, U, V, P, d, W, S);
  REQUIRE(flag == 0);

  // Reconstruct the L matrix
  Matrix UWT;
  celerite2::core2::to_dense(Eigen::VectorXd::Ones(N), U, W, P, UWT);
  UWT.triangularView<Eigen::StrictlyUpper>().setConstant(0.0);

  // Brute force the Cholesky factorization
  Eigen::LDLT<Matrix> LDLT(K);
  Eigen::MatrixXd matrixL = LDLT.matrixL();

  // Check that the lower triangle is correct
  double resid = (matrixL - UWT).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);

  // Check that the diagonal is correct
  double diag_resid = (LDLT.vectorD() - d).array().abs().maxCoeff();
  REQUIRE(diag_resid < 1e-12);
}
