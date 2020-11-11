#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "helpers.hpp"

#include <celerite2/celerite2.h>

using namespace celerite2::test;
using namespace celerite2::core;

TEMPLATE_LIST_TEST_CASE("check the results of to_dense", "[to_dense]", TestKernels) {
  auto kernel = TestType::get_kernel();

  Vector x, diag;
  Matrix Y;
  std::tie(x, diag, Y) = get_data();
  const int N          = x.rows();

  Vector ar, cr, ac, bc, cc, dc;
  std::tie(ar, cr, ac, bc, cc, dc) = kernel.get_coefficients();
  auto matrices                    = kernel.get_celerite_matrices(x, diag);

  int nr = ar.rows(), nc = ac.rows();

  Matrix K;
  to_dense(x, std::get<0>(matrices), std::get<1>(matrices), std::get<2>(matrices), std::get<3>(matrices), K);

  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < N; ++m) {
      auto tau     = std::abs(x(n) - x(m));
      double value = 0.0;
      for (int j = 0; j < nr; ++j) { value += ar(j) * std::exp(-cr(j) * tau); }
      for (int j = 0; j < nc; ++j) {
        auto arg = dc(j) * tau;
        value += std::exp(-cc(j) * tau) * (ac(j) * std::cos(arg) + bc(j) * std::sin(arg));
      }
      if (n == m) { value += diag(n); }
      REQUIRE(std::abs(value - K(n, m)) < 1e-8);
    }
  }
}
