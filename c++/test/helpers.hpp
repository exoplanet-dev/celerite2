#include <tuple>
#include <Eigen/Core>

#include <celerite2/terms.hpp>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

// A helper function for generating test data
std::tuple<Vector, Vector, Matrix> get_data(const int N = 50, const int Nrhs = 5) {
  Vector x(N), diag(N);
  Matrix Y(N, Nrhs);
  for (int i = 0; i < N; ++i) {
    auto delta = double(i) / (N - 1);
    x(i)       = 10 * delta + delta * delta;
    diag(i)    = 0.5;
    for (int j = 0; j < Nrhs; ++j) { Y(i, j) = sin(x(i) + double(j) / Nrhs); }
  }
  return std::make_tuple(x, diag, Y);
}

// Create a list of test kernels
struct TestKernelReal {
  static auto get_kernel() { return celerite2::RealTerm<double>(1.0, 0.1); }
};

struct TestKernelComplex {
  static auto get_kernel() { return celerite2::ComplexTerm<double>(0.8, 0.03, 1.0, 0.1); }
};

struct TestKernelSHO1 {
  static auto get_kernel() { return celerite2::SHOTerm<double>(1.2, 0.3, 0.1); }
};

struct TestKernelSHO2 {
  static auto get_kernel() { return celerite2::SHOTerm<double>(0.1, 1.3, 5.3); }
};

struct TestKernelSum1 {
  static auto get_kernel() { return TestKernelReal::get_kernel() + TestKernelComplex::get_kernel(); }
};

struct TestKernelSum2 {
  static auto get_kernel() { return TestKernelReal::get_kernel() + TestKernelComplex::get_kernel() + TestKernelSHO1::get_kernel(); }
};

struct TestKernelSum3 {
  static auto get_kernel() {
    return TestKernelReal::get_kernel() + TestKernelComplex::get_kernel() + TestKernelSHO1::get_kernel() + TestKernelSHO2::get_kernel();
  }
};

struct TestKernelSum4 {
  static auto get_kernel() { return TestKernelSHO1::get_kernel() + TestKernelSHO2::get_kernel(); }
};

using TestKernels =
   std::tuple<TestKernelReal, TestKernelComplex, TestKernelSHO1, TestKernelSHO2, TestKernelSum1, TestKernelSum2, TestKernelSum3, TestKernelSum4>;
