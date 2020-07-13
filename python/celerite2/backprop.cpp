#include <pybind11/pybind11.h>
#include <cmath>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {

//
// BACKPROP INTERFACE
//
auto factor_fwd(py::array_t<double, py::array::c_style> a, py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V,
                py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W,
                py::array_t<double, py::array::c_style> S) {
  SETUP_BASE_MATRICES;

  GET_BUF_VEC(a, N);
  GET_BUF_MAT(V, N, J);
  GET_BUF_MAT(S, N, J * J);

  Eigen::Index flag = 0;
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(a_, abuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    VECTOR(d_, dbuf, N);                                                                                                                             \
    MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                                    \
    MATRIX((SIZE * SIZE), S_, Sbuf, N, (J * J));                                                                                                     \
    flag = celerite2::core::factor(a_, U_, V_, P_, d_, W_, S_);                                                                                      \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
  if (flag) throw linalg_exception();
  return std::make_tuple(d, W, S);
}

auto factor_rev(py::array_t<double, py::array::c_style> a, py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V,
                py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W,
                py::array_t<double, py::array::c_style> S, py::array_t<double, py::array::c_style> bd, py::array_t<double, py::array::c_style> bW,
                py::array_t<double, py::array::c_style> ba, py::array_t<double, py::array::c_style> bU, py::array_t<double, py::array::c_style> bV,
                py::array_t<double, py::array::c_style> bP) {
  SETUP_BASE_MATRICES;

  GET_BUF_VEC(a, N);
  GET_BUF_MAT(V, N, J);
  GET_BUF_MAT(S, N, J * J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  GET_BUF_VEC(ba, N);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_MAT(bV, N, J);
  GET_BUF_MAT(bP, N - 1, J);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(a_, abuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, P_, Pbuf, N - 1, J);                                                                                                          \
    CONST_VECTOR(d_, dbuf, N);                                                                                                                       \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    CONST_MATRIX((SIZE * SIZE), S_, Sbuf, N, (J * J));                                                                                               \
    CONST_VECTOR(bd_, bdbuf, N);                                                                                                                     \
    CONST_MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                            \
    VECTOR(ba_, babuf, N);                                                                                                                           \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bV_, bVbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bP_, bPbuf, N - 1, J);                                                                                                              \
    celerite2::core::factor_rev(a_, U_, V_, P_, d_, W_, S_, bd_, bW_, ba_, bU_, bV_, bP_);                                                           \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP

  return std::make_tuple(ba, bU, bV, bP);
}

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(backprop, m) {

  py::register_exception<celerite2::driver::linalg_exception>(m, "LinAlgError");

  // Backprop interface
  m.def("factor_fwd", &celerite2::driver::factor_fwd, py::arg("a").noconvert(), py::arg("U").noconvert(), py::arg("V").noconvert(),
        py::arg("P").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("S").noconvert());
  m.def("factor_rev", &celerite2::driver::factor_rev, py::arg("a").noconvert(), py::arg("U").noconvert(), py::arg("V").noconvert(),
        py::arg("P").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("S").noconvert(), py::arg("bd").noconvert(),
        py::arg("bW").noconvert(), py::arg("ba").noconvert(), py::arg("b").noconvert(), py::arg("V").noconvert(), py::arg("bP").noconvert());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
