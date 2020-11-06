#include <pybind11/pybind11.h>
#include <cmath>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {

//
// THE PYBIND11 INTERFACE IMPLEMENTATION
//
auto factor(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
            py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W) {
  SETUP_BASE_MATRICES;
  Eigen::Index flag = 0;
  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(a_, dbuf, N);
  VECTOR(d_, dbuf, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Wbuf, N, J);                                                                                                              \
    MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                                    \
    flag = celerite2::core::factor(t_, c_, a_, U_, V_, d_, W_);                                                                                      \
  }
  UNWRAP_CASES_MOST;
#undef FIXED_SIZE_MAP
  if (flag) throw driver_linalg_exception();
  return std::make_tuple(d, W);
}

auto solve(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
           py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::solve(t_, c_, U_, d_, W_, Z_, Z_);                                                                                            \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::solve(t_, c_, U_, d_, W_, Z_, Z_);                                                                                            \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

auto norm(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
          py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  if (nrhs != 1) throw std::invalid_argument("Z must be a vector");
  Eigen::Matrix<double, 1, 1> norm_;
  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  VECTOR(Z_, Zbuf, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    celerite2::core::norm(t_, c_, U_, d_, W_, Z_, norm_, Z_);                                                                                        \
  }
  UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
  return norm_(0, 0);
}

auto matmul(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> d,
            py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
            py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(Z);
  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Y_, Ybuf, N);                                                                                                                           \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::matmul(t_, c_, d_, U_, W_, Y_, Z_);                                                                                           \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::matmul(t_, c_, d_, U_, W_, Y_, Z_);                                                                                           \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

auto dot_tril(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
              py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      celerite2::core::dot_tril(t_, c_, U_, d_, W_, Z_, Z_);                                                                                         \
    } else {                                                                                                                                         \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      celerite2::core::dot_tril(t_, c_, U_, d_, W_, Z_, Z_);                                                                                         \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

template <bool Lower = true>
struct do_general_dot {
  template <typename A, typename B, typename C, typename D, typename E>
  static void apply(const A &t1, const A &t2, const B &c, const C &U, const C &V, const D &Y, E &Z) {
    celerite2::core::general_lower_dot(t1, t2, c, U, V, Y, Z);
  }
};
template <>
struct do_general_dot<false> {
  template <typename A, typename B, typename C, typename D, typename E>
  static void apply(const A &t1, const A &t2, const B &c, const C &U, const C &V, const D &Y, E &Z) {
    celerite2::core::general_upper_dot(t1, t2, c, U, V, Y, Z);
  }
};

template <bool Lower>
auto general_dot(py::array_t<double, py::array::c_style> t1, py::array_t<double, py::array::c_style> t2, py::array_t<double, py::array::c_style> c,
                 py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V, py::array_t<double, py::array::c_style> Y,
                 py::array_t<double, py::array::c_style> Z) {
  py::buffer_info t1buf = t1.request(), t2buf = t2.request(), cbuf = c.request(), Ubuf = U.request(), Vbuf = V.request();
  if (t1buf.ndim != 1 || t2buf.ndim != 1 || cbuf.ndim != 1 || Ubuf.ndim != 2 || Vbuf.ndim != 2) throw std::invalid_argument("Invalid dimensions");
  ssize_t N = t1buf.shape[0], M = t2buf.shape[0], J = cbuf.shape[0];
  if (N == 0 || M == 0 || J == 0) throw std::invalid_argument("Dimensions can't be zero");
  if (Ubuf.shape[0] != N || Ubuf.shape[1] != J) throw std::invalid_argument("Invalid shape: U");
  if (Vbuf.shape[0] != M || Vbuf.shape[1] != J) throw std::invalid_argument("Invalid shape: V");

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX_WITH_SIZE(Y, M);
  SETUP_RHS_MATRIX_WITH_SIZE(Z, N);
  CONST_VECTOR(t1_, t1buf, N);
  CONST_VECTOR(t2_, t2buf, M);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, M, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, M);                                                                                                                     \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      do_general_dot<Lower>::apply(t1_, t2_, c_, U_, V_, Y_, Z_);                                                                                    \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, M, nrhs);                                                                                               \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      do_general_dot<Lower>::apply(t1_, t2_, c_, U_, V_, Y_, Z_);                                                                                    \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

auto get_celerite_matrices(py::array_t<double, py::array::c_style> ar_in, py::array_t<double, py::array::c_style> ac_in,
                           py::array_t<double, py::array::c_style> bc_in, py::array_t<double, py::array::c_style> dc_in,
                           py::array_t<double, py::array::c_style> x_in, py::array_t<double, py::array::c_style> diag_in,
                           py::array_t<double, py::array::c_style> a_out, py::array_t<double, py::array::c_style> U_out,
                           py::array_t<double, py::array::c_style> V_out) {
  auto ar = ar_in.unchecked<1>();
  auto ac = ac_in.unchecked<1>();
  auto bc = bc_in.unchecked<1>();
  auto dc = dc_in.unchecked<1>();

  auto x    = x_in.unchecked<1>();
  auto diag = diag_in.unchecked<1>();

  auto a = a_out.mutable_unchecked<1>();
  auto U = U_out.mutable_unchecked<2>();
  auto V = V_out.mutable_unchecked<2>();

  ssize_t N = x.shape(0), Jr = ar.shape(0), Jc = ac.shape(0), J = Jr + 2 * Jc;

  if (bc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: bc");
  if (dc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: dc");

  if (diag.shape(0) != N) throw std::invalid_argument("dimension mismatch: diag");

  if (a.shape(0) != N) throw std::invalid_argument("dimension mismatch: a");
  if (U.shape(0) != N || U.shape(1) != J) throw std::invalid_argument("dimension mismatch: U");
  if (V.shape(0) != N || V.shape(1) != J) throw std::invalid_argument("dimension mismatch: V");

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

  return std::make_tuple(a_out, U_out, V_out);
}

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(driver, m) {
  m.doc() = R"doc(
    The computation engine for celerite2

    These functions are low level and you shouldn't generally need or want to call them as a user.
)doc";

  py::register_exception<celerite2::driver::driver_linalg_exception>(m, "LinAlgError");

  m.def("factor", &celerite2::driver::factor, "Compute the Cholesky factor of a celerite system");
  m.def("solve", &celerite2::driver::solve, "Solve a celerite system using the output of `factor`");
  m.def("norm", &celerite2::driver::norm, "Compute the norm of a celerite system applied to a vector");
  m.def("matmul", &celerite2::driver::matmul, "Dot a celerite system into a matrix or vector");
  m.def("dot_tril", &celerite2::driver::dot_tril, "Dot the Cholesky factor celerite system into a matrix or vector");
  m.def("general_lower_dot", &celerite2::driver::general_dot<true>, "The general lower-triangular dot product of a rectangular celerite system");
  m.def("general_upper_dot", &celerite2::driver::general_dot<false>, "The general upper-triangular dot product of a rectangular celerite system");

  m.def("get_celerite_matrices", &celerite2::driver::get_celerite_matrices, "Get the matrices defined by a celerite system",
        py::arg("ar").noconvert(), py::arg("ac").noconvert(), py::arg("bc").noconvert(), py::arg("dc").noconvert(), py::arg("x").noconvert(),
        py::arg("diag").noconvert(), py::arg("a").noconvert(), py::arg("U").noconvert(), py::arg("W").noconvert());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
