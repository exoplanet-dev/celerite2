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
  UNWRAP_CASES_MOST;
#undef FIXED_SIZE_MAP
  if (flag) throw driver_linalg_exception();
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
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

auto norm(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
          py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {
  SETUP_BASE_MATRICES;
  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  if (nrhs != 1) throw std::invalid_argument("Z must be a vector");
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
  UNWRAP_CASES_MOST
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
  UNWRAP_CASES_FEW
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
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return Z;
}

auto conditional_mean(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V, py::array_t<double, py::array::c_style> P,
                      py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> U_star,
                      py::array_t<double, py::array::c_style> V_star, py::array_t<ssize_t, py::array::c_style> inds,
                      py::array_t<double, py::array::c_style> mu) {
  py::buffer_info Ubuf = U.request();
  py::buffer_info Vbuf = V.request();
  py::buffer_info Pbuf = P.request();
  if (Ubuf.ndim != 2 || Vbuf.ndim != 2 || Pbuf.ndim != 2) throw std::invalid_argument("Invalid dimensions");
  ssize_t N = Ubuf.shape[0], J = Ubuf.shape[1];
  if (N == 0 || J == 0) throw std::invalid_argument("Dimensions can't be zero");
  if (Vbuf.shape[0] != N || Vbuf.shape[1] != J) throw std::invalid_argument("Invalid shape: W");
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::invalid_argument("Invalid shape: P");

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  if (nrhs != 1) throw std::invalid_argument("Z must be a vector");

  py::buffer_info indsbuf = inds.request();
  if (indsbuf.ndim != 1) throw std::invalid_argument("Invalid shape: inds");
  ssize_t M = indsbuf.shape[0];

  GET_BUF_MAT(U_star, M, J);
  GET_BUF_MAT(V_star, M, J);
  GET_BUF_VEC(mu, M);

  CONST_VECTOR(Z_, Zbuf, N);
  VECTOR(mu_, mubuf, M);
  Eigen::Map<Eigen::Matrix<ssize_t, Eigen::Dynamic, 1>> inds_((ssize_t *)indsbuf.ptr, M, 1);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_MATRIX(SIZE, U_star_, U_starbuf, M, J);                                                                                                    \
    CONST_MATRIX(SIZE, V_star_, V_starbuf, M, J);                                                                                                    \
    celerite2::core::conditional_mean(U_, V_, P_, Z_, U_star_, V_star_, inds_, mu_);                                                                 \
  }
  UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
  return mu;
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

  if (cr.shape(0) != Jr) throw std::invalid_argument("dimension mismatch: cr");
  if (bc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: bc");
  if (cc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: cc");
  if (dc.shape(0) != Jc) throw std::invalid_argument("dimension mismatch: dc");

  if (diag.shape(0) != N) throw std::invalid_argument("dimension mismatch: diag");

  if (a.shape(0) != N) throw std::invalid_argument("dimension mismatch: a");
  if (U.shape(0) != N || U.shape(1) != J) throw std::invalid_argument("dimension mismatch: U");
  if (V.shape(0) != N || V.shape(1) != J) throw std::invalid_argument("dimension mismatch: V");
  if (P.shape(0) != N - 1 || P.shape(1) != J) throw std::invalid_argument("dimension mismatch: P");

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

  py::register_exception<celerite2::driver::driver_linalg_exception>(m, "LinAlgError");

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
  m.def("conditional_mean", &celerite2::driver::conditional_mean, "Copmpute the conditional mean", py::arg("U").noconvert(), py::arg("V").noconvert(),
        py::arg("P").noconvert(), py::arg("Z").noconvert(), py::arg("U_star").noconvert(), py::arg("V_star").noconvert(), py::arg("inds").noconvert(),
        py::arg("mu").noconvert());

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
