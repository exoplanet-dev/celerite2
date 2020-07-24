#ifndef _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_
#define _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>

#include <celerite2/celerite2.h>

namespace celerite2 {
namespace driver {

namespace py = pybind11;

struct driver_linalg_exception : public std::exception {
  const char *what() const throw() { return "failed to factorize or solve matrix"; }
};

struct backprop_linalg_exception : public std::exception {
  const char *what() const throw() { return "failed to factorize or solve matrix"; }
};

//
// SOME USEFUL MACROS
//

// Some pre-processor magic to get faster runtimes for small systems

#define UNWRAP_CASES_FEW                                                                                                                             \
  switch (J) {                                                                                                                                       \
    case 1: FIXED_SIZE_MAP(1); break;                                                                                                                \
    case 2: FIXED_SIZE_MAP(2); break;                                                                                                                \
    case 3: FIXED_SIZE_MAP(3); break;                                                                                                                \
    case 4: FIXED_SIZE_MAP(4); break;                                                                                                                \
    case 5: FIXED_SIZE_MAP(5); break;                                                                                                                \
    case 6: FIXED_SIZE_MAP(6); break;                                                                                                                \
    case 7: FIXED_SIZE_MAP(7); break;                                                                                                                \
    case 8: FIXED_SIZE_MAP(8); break;                                                                                                                \
    case 9: FIXED_SIZE_MAP(9); break;                                                                                                                \
    case 10: FIXED_SIZE_MAP(10); break;                                                                                                              \
    default: FIXED_SIZE_MAP(Eigen::Dynamic);                                                                                                         \
  }

#ifdef CELERITE2_FAST_RUN

#define UNWRAP_CASES                                                                                                                                 \
  switch (J) {                                                                                                                                       \
    case 1: FIXED_SIZE_MAP(1); break;                                                                                                                \
    case 2: FIXED_SIZE_MAP(2); break;                                                                                                                \
    case 3: FIXED_SIZE_MAP(3); break;                                                                                                                \
    case 4: FIXED_SIZE_MAP(4); break;                                                                                                                \
    case 5: FIXED_SIZE_MAP(5); break;                                                                                                                \
    case 6: FIXED_SIZE_MAP(6); break;                                                                                                                \
    case 7: FIXED_SIZE_MAP(7); break;                                                                                                                \
    case 8: FIXED_SIZE_MAP(8); break;                                                                                                                \
    case 9: FIXED_SIZE_MAP(9); break;                                                                                                                \
    case 10: FIXED_SIZE_MAP(10); break;                                                                                                              \
    case 11: FIXED_SIZE_MAP(11); break;                                                                                                              \
    case 12: FIXED_SIZE_MAP(12); break;                                                                                                              \
    case 13: FIXED_SIZE_MAP(13); break;                                                                                                              \
    case 14: FIXED_SIZE_MAP(14); break;                                                                                                              \
    case 15: FIXED_SIZE_MAP(15); break;                                                                                                              \
    case 16: FIXED_SIZE_MAP(16); break;                                                                                                              \
    default: FIXED_SIZE_MAP(Eigen::Dynamic);                                                                                                         \
  }

#define UNWRAP_CASES_MOST                                                                                                                            \
  switch (J) {                                                                                                                                       \
    case 1: FIXED_SIZE_MAP(1); break;                                                                                                                \
    case 2: FIXED_SIZE_MAP(2); break;                                                                                                                \
    case 3: FIXED_SIZE_MAP(3); break;                                                                                                                \
    case 4: FIXED_SIZE_MAP(4); break;                                                                                                                \
    case 5: FIXED_SIZE_MAP(5); break;                                                                                                                \
    case 6: FIXED_SIZE_MAP(6); break;                                                                                                                \
    case 7: FIXED_SIZE_MAP(7); break;                                                                                                                \
    case 8: FIXED_SIZE_MAP(8); break;                                                                                                                \
    case 9: FIXED_SIZE_MAP(9); break;                                                                                                                \
    case 10: FIXED_SIZE_MAP(10); break;                                                                                                              \
    case 11: FIXED_SIZE_MAP(11); break;                                                                                                              \
    case 12: FIXED_SIZE_MAP(12); break;                                                                                                              \
    case 13: FIXED_SIZE_MAP(13); break;                                                                                                              \
    case 14: FIXED_SIZE_MAP(14); break;                                                                                                              \
    case 15: FIXED_SIZE_MAP(15); break;                                                                                                              \
    case 16: FIXED_SIZE_MAP(16); break;                                                                                                              \
    case 17: FIXED_SIZE_MAP(17); break;                                                                                                              \
    case 18: FIXED_SIZE_MAP(18); break;                                                                                                              \
    case 19: FIXED_SIZE_MAP(19); break;                                                                                                              \
    case 20: FIXED_SIZE_MAP(20); break;                                                                                                              \
    case 21: FIXED_SIZE_MAP(21); break;                                                                                                              \
    case 22: FIXED_SIZE_MAP(22); break;                                                                                                              \
    case 23: FIXED_SIZE_MAP(23); break;                                                                                                              \
    case 24: FIXED_SIZE_MAP(24); break;                                                                                                              \
    case 25: FIXED_SIZE_MAP(25); break;                                                                                                              \
    case 26: FIXED_SIZE_MAP(26); break;                                                                                                              \
    case 27: FIXED_SIZE_MAP(27); break;                                                                                                              \
    case 28: FIXED_SIZE_MAP(28); break;                                                                                                              \
    case 29: FIXED_SIZE_MAP(29); break;                                                                                                              \
    case 30: FIXED_SIZE_MAP(30); break;                                                                                                              \
    case 31: FIXED_SIZE_MAP(31); break;                                                                                                              \
    case 32: FIXED_SIZE_MAP(32); break;                                                                                                              \
    default: FIXED_SIZE_MAP(Eigen::Dynamic);                                                                                                         \
  }

#else

#define UNWRAP_CASES UNWRAP_CASES_FEW
#define UNWRAP_CASES_MOST UNWRAP_CASES_FEW

#endif

// These are some generally useful macros for interfacing between numpy and Eigen
template <int Size>
struct order {
  const static int value = Eigen::RowMajor;
};

template <>
struct order<1> {
  const static int value = Eigen::ColMajor;
};

#define VECTOR(NAME, BUF, ROWS) Eigen::Map<Eigen::VectorXd> NAME((double *)BUF.ptr, ROWS, 1)
#define MATRIX(SIZE, NAME, BUF, ROWS, COLS)                                                                                                          \
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> NAME((double *)BUF.ptr, ROWS, COLS)
#define CONST_VECTOR(NAME, BUF, ROWS) Eigen::Map<const Eigen::VectorXd> NAME((double *)BUF.ptr, ROWS, 1)
#define CONST_MATRIX(SIZE, NAME, BUF, ROWS, COLS)                                                                                                    \
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> NAME((double *)BUF.ptr, ROWS, COLS)

#define GET_BUF(NAME, SIZE)                                                                                                                          \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  if (NAME##buf.size != SIZE) throw std::invalid_argument("Invalid shape: " #NAME);
#define GET_BUF_VEC(NAME, ROWS)                                                                                                                      \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  if (NAME##buf.ndim != 1 || NAME##buf.shape[0] != ROWS) throw std::invalid_argument("Invalid shape: " #NAME);
#define GET_BUF_MAT(NAME, ROWS, COLS)                                                                                                                \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  if (NAME##buf.ndim != 2 || NAME##buf.shape[0] != ROWS || NAME##buf.shape[1] != COLS) throw std::invalid_argument("Invalid shape: " #NAME);

// This gets the buffer info for the standard celerite matrix inputs and checks the dimensions
#define SETUP_BASE_MATRICES                                                                                                                          \
  py::buffer_info Ubuf = U.request(), Pbuf = P.request(), dbuf = d.request(), Wbuf = W.request();                                                    \
  if (Ubuf.ndim != 2 || Pbuf.ndim != 2 || dbuf.ndim != 1 || Wbuf.ndim != 2) throw std::invalid_argument("Invalid dimensions");                       \
  ssize_t N = Ubuf.shape[0], J = Ubuf.shape[1];                                                                                                      \
  if (N == 0 || J == 0) throw std::invalid_argument("Dimensions can't be zero");                                                                     \
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::invalid_argument("Invalid shape: P");                                                 \
  if (dbuf.shape[0] != N) throw std::invalid_argument("Invalid shape: d");                                                                           \
  if (Wbuf.shape[0] != N || Wbuf.shape[1] != J) throw std::invalid_argument("Invalid shape: W");

// This gets the buffer info for a right hand side input and checks the dimensions
#define SETUP_RHS_MATRIX(NAME)                                                                                                                       \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  ssize_t NAME##_nrhs       = 1;                                                                                                                     \
  if (NAME##buf.ndim == 2) {                                                                                                                         \
    NAME##_nrhs = NAME##buf.shape[1];                                                                                                                \
  } else if (NAME##buf.ndim != 1)                                                                                                                    \
    throw std::invalid_argument(#NAME " must be a matrix");                                                                                          \
  if (NAME##buf.shape[0] != N) throw std::invalid_argument("Invalid shape: " #NAME);                                                                 \
  if (nrhs > 0 && nrhs != NAME##_nrhs) {                                                                                                             \
    throw std::invalid_argument("dimension mismatch: " #NAME);                                                                                       \
  } else {                                                                                                                                           \
    nrhs = NAME##_nrhs;                                                                                                                              \
  }

};     // namespace driver
};     // namespace celerite2
#endif // _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_
