#include <pybind11/pybind11.h>
#include <cmath>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {

//
// THE PYBIND11 INTERFACE IMPLEMENTATION
//
auto factor(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
            py::array_t<double, py::array::c_style> W) {
  SETUP_BASE_MATRICES;
  Eigen::Index flag = 0;
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(a_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Wbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    VECTOR(d_, dbuf, N);                                                                                                                             \
    MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                                    \
    flag = celerite2::core::factor(a_, U_, V_, P_, d_, W_);                                                                                          \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
  if (flag) throw linalg_exception();
  return std::make_tuple(d, W);
}

auto solve(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
           py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_VECTOR(d_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::solve(U_, P_, d_, W_, Z_, Z_);                                                                                                \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::solve(U_, P_, d_, W_, Z_, Z_);                                                                                                \
    }                                                                                                                                                \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP
  return Z;
}

auto norm(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
          py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  if (nrhs != 1) throw std::runtime_error("Z must be a vector");
  Eigen::Matrix<double, 1, 1> norm_;
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_VECTOR(d_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    VECTOR(Z_, Zbuf, N);                                                                                                                             \
    celerite2::core::norm(U_, P_, d_, W_, Z_, norm_, Z_);                                                                                            \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP
  return norm_(0, 0);
}

auto matmul(py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> W,
            py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> Y, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_VECTOR(d_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Y_, Ybuf, N);                                                                                                                           \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::matmul(d_, U_, W_, P_, Y_, Z_);                                                                                               \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::matmul(d_, U_, W_, P_, Y_, Z_);                                                                                               \
    }                                                                                                                                                \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP
  return Z;
}

auto dot_tril(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
              py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_VECTOR(d_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::dot_tril(U_, P_, d_, W_, Z_, Z_);                                                                                             \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::dot_tril(U_, P_, d_, W_, Z_, Z_);                                                                                             \
    }                                                                                                                                                \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP
  return Z;
}

auto get_celerite_matrices(py::array_t<double, py::array::c_style> ar_in, py::array_t<double, py::array::c_style> cr_in,
                           py::array_t<double, py::array::c_style> ac_in, py::array_t<double, py::array::c_style> bc_in,
                           py::array_t<double, py::array::c_style> cc_in, py::array_t<double, py::array::c_style> dc_in,
                           py::array_t<double, py::array::c_style> x_in, py::array_t<double, py::array::c_style> diag_in,
                           py::array_t<double, py::array::c_style> a_out, py::array_t<double, py::array::c_style> U_out,
                           py::array_t<double, py::array::c_style> V_out, py::array_t<double, py::array::c_style> P_out) {
  auto ar = ar_in.unchecked<1>();
  auto cr = cr_in.unchecked<1>();
  auto ac = ac_in.unchecked<1>();
  auto bc = bc_in.unchecked<1>();
  auto cc = cc_in.unchecked<1>();
  auto dc = dc_in.unchecked<1>();

  auto x    = x_in.unchecked<1>();
  auto diag = diag_in.unchecked<1>();

  auto a = a_out.mutable_unchecked<1>();
  auto U = U_out.mutable_unchecked<2>();
  auto V = V_out.mutable_unchecked<2>();
  auto P = P_out.mutable_unchecked<2>();

  ssize_t N = x.shape(0), Jr = ar.shape(0), Jc = ac.shape(0), J = Jr + 2 * Jc;

  if (cr.shape(0) != Jr) throw std::runtime_error("dimension mismatch: cr");
  if (bc.shape(0) != Jc) throw std::runtime_error("dimension mismatch: bc");
  if (cc.shape(0) != Jc) throw std::runtime_error("dimension mismatch: cc");
  if (dc.shape(0) != Jc) throw std::runtime_error("dimension mismatch: dc");

  if (diag.shape(0) != N) throw std::runtime_error("dimension mismatch: diag");

  if (a.shape(0) != N) throw std::runtime_error("dimension mismatch: a");
  if (U.shape(0) != N || U.shape(1) != J) throw std::runtime_error("dimension mismatch: U");
  if (V.shape(0) != N || V.shape(1) != J) throw std::runtime_error("dimension mismatch: V");
  if (P.shape(0) != N - 1 || P.shape(1) != J) throw std::runtime_error("dimension mismatch: P");

  double sum = 0.0;
  for (ssize_t j = 0; j < Jr; ++j) sum += ar(j);
  for (ssize_t j = 0; j < Jc; ++j) sum += ac(j);

  for (ssize_t n = 0; n < N; ++n) {
    a(n) = diag(n) + sum;
    for (ssize_t j = 0; j < Jr; ++j) {
      V(n, j) = 1.0;
      U(n, j) = ar(j);
    }
    for (ssize_t j = 0, ind = Jr; j < Jc; ++j, ind += 2) {
      double arg = dc(j) * x(n);
      double cos = V(n, ind) = std::cos(arg);
      double sin = V(n, ind + 1) = std::sin(arg);

      U(n, ind)     = ac(j) * cos + bc(j) * sin;
      U(n, ind + 1) = ac(j) * sin - bc(j) * cos;
    }
  }

  for (ssize_t n = 0; n < N - 1; ++n) {
    double dx = x(n + 1) - x(n);
    for (ssize_t j = 0; j < Jr; ++j) P(n, j) = std::exp(-cr(j) * dx);
    for (ssize_t j = 0, ind = Jr; j < Jc; ++j, ind += 2) P(n, ind) = P(n, ind + 1) = std::exp(-cc(j) * dx);
  }

  return std::make_tuple(a_out, U_out, V_out, P_out);
}

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(driver, m) {
  m.doc() = R"doc(
    The computation engine for celerite2

    These functions are low level and you shouldn't generally need or want to call them as a user.
)doc";

  py::register_exception<celerite2::driver::linalg_exception>(m, "LinAlgError");

  m.def("factor", &celerite2::driver::factor, "Compute the Cholesky factor of a celerite system", py::arg("U").noconvert(), py::arg("P").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert());
  m.def("solve", &celerite2::driver::solve, "Solve a celerite system using the output of `factor`", py::arg("U").noconvert(),
        py::arg("P").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Z").noconvert());
  m.def("norm", &celerite2::driver::norm, "Compute the norm of a celerite system applied to a vector", py::arg("U").noconvert(),
        py::arg("P").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("z").noconvert());
  m.def("matmul", &celerite2::driver::matmul, "Dot a celerite system into a matrix or vector", py::arg("a").noconvert(), py::arg("U").noconvert(),
        py::arg("W").noconvert(), py::arg("P").noconvert(), py::arg("Y").noconvert(), py::arg("Z").noconvert());
  m.def("dot_tril", &celerite2::driver::dot_tril, "Dot the Cholesky factor celerite system into a matrix or vector", py::arg("U").noconvert(),
        py::arg("P").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Z").noconvert());

  m.def("get_celerite_matrices", &celerite2::driver::get_celerite_matrices, "Get the matrices defined by a celerite system",
        py::arg("ar").noconvert(), py::arg("cr").noconvert(), py::arg("ac").noconvert(), py::arg("bc").noconvert(), py::arg("cc").noconvert(),
        py::arg("dc").noconvert(), py::arg("x").noconvert(), py::arg("diag").noconvert(), py::arg("a").noconvert(), py::arg("U").noconvert(),
        py::arg("W").noconvert(), py::arg("P").noconvert());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
