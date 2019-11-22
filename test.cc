#include <iostream>
#include <Eigen/Core>
#include "terms.hpp"

int main() {
  const int N = 500;
  Eigen::VectorXd x(N), diag(N);

  for (int i = 0; i < N; ++i) {
    x(i)    = 10 * double(i) / (N - 1);
    diag(i) = 0.01;
  }

  celerite::SHOTerm<double> kernel1(0.2, 5.6, 2.3);
  celerite::SHOTerm<double> kernel2(1.2, 0.3, 0.1);
  celerite::SHOTerm<double> kernel3(0.1, 1.3, 5.3);
  auto kernel = kernel1 + kernel2 + kernel3;

  const auto &[a, U, V, P] = kernel.get_celerite_matrices(x, diag);

  std::cout << P << "\n";

  return 0;
}