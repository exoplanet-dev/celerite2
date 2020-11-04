#include <tuple>
#include <vector>
#include <Eigen/Core>
#include <celerite2/terms.hpp>
#include <celerite2/internal.hpp>

namespace celerite2 {
namespace test {

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

//
// UNIT TESTING FOR GRADIENTS
//
// The basic procedure is as follows:
//
// 1. Compute the numerical Jacobians using a first order forward diff
// 2. Loop over entries in each output and compute the reverse sweep; compare to the numerical estimate
//
// The `jacobian` object is a `vector` of `vector`s of `MatrixXd` representations of the Jacobians. This will have shape:
//
// `jacobian[number_of_inputs][number_of_outputs][size_of_input, size_of_output]`
//

// Loop over output indices and save the numerically estimated gradient of that output with repect to a given input parameter
template <int OutputNumber, int NumberOfOutputs>
struct save_jacobian {
  template <typename Value>
  void operator()(Eigen::Index input_number, Eigen::Index size_of_input, Eigen::Index index_in_input, double eps, const Value value0,
                  const Value value, std::vector<std::vector<Eigen::MatrixXd>> &jacobian) const {
    if constexpr (OutputNumber < NumberOfOutputs) {
      // Compute the numerical derivative and a flat view
      typedef typename std::tuple_element<OutputNumber, Value>::type MatType;
      Eigen::Matrix<double, MatType::RowsAtCompileTime, MatType::ColsAtCompileTime, MatType::IsRowMajor> grad =
         (std::get<OutputNumber>(value) - std::get<OutputNumber>(value0)) / eps;

      // Resize the Jacobian
      if (index_in_input == 0) { jacobian[input_number][OutputNumber].resize(size_of_input, grad.size()); }

      // Copy from a flat view of the Jacobian into the container
      jacobian[input_number][OutputNumber].row(index_in_input) = Eigen::Map<Eigen::Matrix<double, 1, Eigen::Dynamic>>(grad.data(), grad.size());

      // Recursively apply this function to the next output
      save_jacobian<OutputNumber + 1, NumberOfOutputs>()(input_number, size_of_input, index_in_input, eps, value0, value, jacobian);
    } else {
      UNUSED(input_number);
      UNUSED(size_of_input);
      UNUSED(index_in_input);
      UNUSED(eps);
      UNUSED(value0);
      UNUSED(value);
      UNUSED(jacobian);
    }
  }
};

// Loop over each input parameter and compute the numerical derivative of each output with respect to this input
template <int InputNumber, int NumberOfInputs>
struct compute_jacobian {
  template <typename Func, typename Args, typename Value>
  void operator()(double eps, Func &&func, Args args, const Value value0, std::vector<std::vector<Eigen::MatrixXd>> &jacobian) const {
    if constexpr (InputNumber < NumberOfInputs) {

      // Extract the current input and construct a flattened view
      auto arg = std::get<InputNumber>(args);
      Eigen::Map<Eigen::VectorXd> arg_map(arg.data(), arg.size());

      // Loop over elements of the input and compute the first difference
      jacobian[InputNumber].resize(std::tuple_size<Value>::value);
      for (Eigen::Index n = 0; n < arg.size(); ++n) {
        arg_map(n) += eps;
        std::get<InputNumber>(args) = arg;
        auto value                  = std::apply(func, args);
        arg_map(n) -= eps;
        std::get<InputNumber>(args) = arg;

        // Loop over outputs and save the approximate gradients
        save_jacobian<0, std::tuple_size<Value>::value>()(InputNumber, arg.size(), n, eps, value0, value, jacobian);
      }

      // Recursively apply this function for the next input
      compute_jacobian<InputNumber + 1, NumberOfInputs>()(eps, func, args, value0, jacobian);
    } else {
      UNUSED(eps);
      UNUSED(func);
      UNUSED(args);
      UNUSED(value0);
      UNUSED(jacobian);
    }
  }
};

// Loop over inputs and call the `setZero` method on each one
template <int Current, int End>
struct zero_args {
  template <typename T>
  void operator()(T &args) const {
    if constexpr (Current < End) {
      std::get<Current>(args).setZero();
      zero_args<Current + 1, End>()(args);
    } else {
      UNUSED(args);
    }
  }
};

template <int InputNumber, int NumberOfInputs>
struct check_rev {
  template <typename RevOut>
  bool operator()(double tol, Eigen::Index output_number, Eigen::Index index_in_output, const RevOut rev_out,
                  const std::vector<std::vector<Eigen::MatrixXd>> &jacobian) const {
    if constexpr (InputNumber < NumberOfInputs) {
      auto arg = std::get<InputNumber>(rev_out);
      Eigen::Map<Eigen::VectorXd> arg_map(arg.data(), arg.size());
      double resid = (jacobian[InputNumber][output_number].col(index_in_output) - arg_map).array().abs().maxCoeff();

      if (resid > tol) {
        std::cerr << "Invalid gradient for input " << InputNumber << " and output " << output_number << "[" << index_in_output
                  << "]; expected:\n\n  >> ";
        std::cerr << jacobian[InputNumber][output_number].col(index_in_output).transpose() << "\n\ngot:\n\n  >> ";
        std::cerr << arg_map.transpose() << "\n\nmax difference: " << resid << "\n";
        return false;
      }

      return check_rev<InputNumber + 1, NumberOfInputs>()(tol, output_number, index_in_output, rev_out, jacobian);
    } else {
      UNUSED(tol);
      UNUSED(output_number);
      UNUSED(index_in_output);
      UNUSED(rev_out);
      UNUSED(jacobian);
    }
    return true;
  }
};

// Loop over outputs and apply the reverse sweep for each element
template <int OutputNumber, int NumberOfOutputs>
struct compute_and_check_rev {
  template <typename Rev, typename Args, typename RevIn, typename RevOut>
  bool operator()(double tol, Rev &&rev, const Args args, RevIn rev_in, RevOut rev_out,
                  const std::vector<std::vector<Eigen::MatrixXd>> &jacobian) const {
    if constexpr (OutputNumber < NumberOfOutputs) {
      // Zero the initial "barred" values
      zero_args<0, std::tuple_size<RevIn>::value>()(rev_in);
      zero_args<0, std::tuple_size<RevOut>::value>()(rev_out);

      // Extract the current "barred" output and construct a flattened view
      auto arg = std::get<OutputNumber>(rev_in);
      Eigen::Map<Eigen::VectorXd> arg_map(arg.data(), arg.size());

      // Loop over elements of the output
      for (Eigen::Index n = 0; n < arg.size(); ++n) {
        arg_map(n)                     = 1.0;
        std::get<OutputNumber>(rev_in) = arg;
        auto value                     = std::apply(rev, std::tuple_cat(args, rev_in, rev_out));
        arg_map(n)                     = 0.0;
        std::get<OutputNumber>(rev_in) = arg;

        // Check the derivatives of this value with respect to all the inputs
        if (!check_rev<0, std::tuple_size<RevOut>::value>()(tol, OutputNumber, n, value, jacobian)) { return false; }
      }

      // Move on to the next output
      return compute_and_check_rev<OutputNumber + 1, NumberOfOutputs>()(tol, rev, args, rev_in, rev_out, jacobian);
    } else {
      UNUSED(tol);
      UNUSED(rev);
      UNUSED(args);
      UNUSED(rev_in);
      UNUSED(rev_out);
      UNUSED(jacobian);
    }
    return true;
  }
};

template <typename Func, typename Rev, typename ArgsIn, typename ArgsExtra, typename RevIn, typename RevOut>
bool check_grad(Func &&func, Rev &&rev, ArgsIn args_in, ArgsExtra args_extra, RevIn rev_in, RevOut rev_out) {
  const double eps = 1.234e-8;
  const double tol = 500 * eps;

  // Compute a reference value for the function
  auto value0 = std::apply(func, std::tuple_cat(args_in, args_extra));

  // Static loop over input arguments to compute the numerical gradient
  std::vector<std::vector<Eigen::MatrixXd>> jacobian(std::tuple_size<ArgsIn>::value);
  compute_jacobian<0, std::tuple_size<ArgsIn>::value>()(eps, func, std::tuple_cat(args_in, args_extra), value0, jacobian);

  // Check the backwards pass
  return compute_and_check_rev<0, std::tuple_size<RevIn>::value>()(tol, rev, std::tuple_cat(args_in, args_extra), rev_in, rev_out, jacobian);
}

#define SETUP_TEST(NUM)                                                                                                                              \
  auto kernel = TestType::get_kernel();                                                                                                              \
  typedef typename decltype(kernel)::LowRank LowRank;                                                                                                \
  typedef typename decltype(kernel)::CoeffVector CoeffVector;                                                                                        \
                                                                                                                                                     \
  /* DATA */                                                                                                                                         \
  Vector x, diag;                                                                                                                                    \
  Matrix Y;                                                                                                                                          \
  std::tie(x, diag, Y)    = get_data(NUM);                                                                                                           \
  const Eigen::Index N    = x.rows();                                                                                                                \
  const Eigen::Index nrhs = Y.cols();                                                                                                                \
                                                                                                                                                     \
  /* CELERITE MATRICES */                                                                                                                            \
  Vector a;                                                                                                                                          \
  CoeffVector c;                                                                                                                                     \
  LowRank U, V;                                                                                                                                      \
  std::tie(c, a, U, V) = kernel.get_celerite_matrices(x, diag);                                                                                      \
  const Eigen::Index J = U.cols();                                                                                                                   \
                                                                                                                                                     \
  UNUSED(N);                                                                                                                                         \
  UNUSED(J);                                                                                                                                         \
  UNUSED(nrhs);

} // namespace test
} // namespace celerite2
