#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <Eigen/Dense>
#include <celerite2/core.hpp>
#include <celerite2/core2.hpp>

TEMPLATE_LIST_TEST_CASE("check the results of solve", "[solve]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix X, Y, Z;
  std::tie(x, diag, Y) = get_data();

  X.resize(Y.rows(), Y.cols());
  Z.resize(Y.rows(), Y.cols());

  typename decltype(kernel)::Vector a, d;
  typename decltype(kernel)::LowRank U, V, P, W;
  std::tie(a, U, V, P) = kernel.get_celerite_matrices(x, diag);

  d.resize(a.rows());
  W.resize(U.rows(), U.cols());

  Matrix K, S, F, G;
  celerite2::core::to_dense(a, U, V, P, K);

  // Do the solve using celerite
  int flag = celerite2::core2::factor(a, U, V, P, d, W, S);
  REQUIRE(flag == 0);
  celerite2::core2::solve(U, P, d, W, Y, X, Z, F, G);

  // Brute force the solve
  Eigen::LDLT<Matrix> LDLT(K);
  Matrix expect = LDLT.solve(Y);

  // Check the result
  double resid = (expect - X).array().abs().maxCoeff();
  REQUIRE(resid < 1e-12);
}
