#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <celerite2/celerite2.h>

#include <iostream>

namespace py = pybind11;

const void factor(void *out_tuple, const void **in) {
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;

  void **out    = reinterpret_cast<void **>(out_tuple);
  int *flag     = reinterpret_cast<int *>(out[0]);
  double *d_out = reinterpret_cast<double *>(out[1]);
  double *W_out = reinterpret_cast<double *>(out[2]);
  double *S_out = reinterpret_cast<double *>(out[3]);

  const int N        = *reinterpret_cast<const int *>(in[0]);
  const int J        = *reinterpret_cast<const int *>(in[1]);
  const double *a_in = reinterpret_cast<const double *>(in[2]);
  const double *U_in = reinterpret_cast<const double *>(in[3]);
  const double *V_in = reinterpret_cast<const double *>(in[4]);
  const double *P_in = reinterpret_cast<const double *>(in[5]);

  Eigen::Map<const Eigen::VectorXd> a(a_in, N);
  Eigen::Map<const Matrix> U(U_in, N, J);
  Eigen::Map<const Matrix> V(V_in, N, J);
  Eigen::Map<const Matrix> P(P_in, N - 1, J);
  Eigen::Map<Eigen::VectorXd> d(d_out, N);
  Eigen::Map<Matrix> W(W_out, N, J);
  Eigen::Map<Matrix> S(S_out, N, J * J);

  celerite2::core::factor(a, U, V, P, d, W, S);
}

PYBIND11_MODULE(xla_ops, m) {
  m.def("factor", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&factor, name);
  });
}
