#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>

#include <celerite2/celerite2.h>

namespace py = pybind11;

namespace celerite2 {
namespace driver {

struct linalg_exception : public std::exception {
  const char *what() const throw() { return "failed to factorize or solve matrix"; }
};

//
// SOME USEFUL MACROS
//

// Some pre-processor magic to get faster runtimes for small systems
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

// These are some generally useful macros for interfacing between numpy and Eigen
#define GET_ROW_MAJOR(SIZE) constexpr int RowMajor = (SIZE == 1) ? Eigen::ColMajor : Eigen::RowMajor
#define VECTOR(NAME, BUF, ROWS) Eigen::Map<Eigen::VectorXd> NAME((double *)BUF.ptr, ROWS, 1)
#define MATRIX(SIZE, NAME, BUF, ROWS, COLS) Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, RowMajor>> NAME((double *)BUF.ptr, ROWS, COLS)
#define CONST_VECTOR(NAME, BUF, ROWS) Eigen::Map<const Eigen::VectorXd> NAME((double *)BUF.ptr, ROWS, 1)
#define CONST_MATRIX(SIZE, NAME, BUF, ROWS, COLS)                                                                                                    \
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, RowMajor>> NAME((double *)BUF.ptr, ROWS, COLS)

// This gets the buffer info for the standard celerite matrix inputs and checks the dimensions
#define SETUP_BASE_MATRICES                                                                                                                          \
  py::buffer_info Ubuf = U.request(), Pbuf = P.request(), dbuf = d.request(), Wbuf = W.request();                                                    \
  if (Ubuf.ndim != 2 || Pbuf.ndim != 2 || dbuf.ndim != 1 || Wbuf.ndim != 2) throw std::runtime_error("Invalid dimensions");                          \
  py::size_t N = Ubuf.shape[0], J = Ubuf.shape[1];                                                                                                   \
  if (N == 0 || J == 0) throw std::runtime_error("Dimensions can't be zero");                                                                        \
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::runtime_error("Invalid shape: P");                                                    \
  if (dbuf.shape[0] != N) throw std::runtime_error("Invalid shape: d");                                                                              \
  if (Wbuf.shape[0] != N || Wbuf.shape[1] != J) throw std::runtime_error("Invalid shape: W");

// This gets the buffer info for a right hand side input and checks the dimensions
#define SETUP_RHS_MATRIX(NAME)                                                                                                                       \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  py::size_t NAME##_nrhs    = 1;                                                                                                                     \
  if (NAME##buf.ndim == 2) {                                                                                                                         \
    NAME##_nrhs = NAME##buf.shape[1];                                                                                                                \
  } else if (NAME##buf.ndim != 1)                                                                                                                    \
    throw std::runtime_error(#NAME " must be a matrix");                                                                                             \
  if (NAME##buf.shape[0] != N) throw std::runtime_error("Invalid shape: " #NAME);                                                                    \
  if (nrhs > 0 && nrhs != NAME##_nrhs) {                                                                                                             \
    throw std::runtime_error("dimension mismatch: " #NAME);                                                                                          \
  } else {                                                                                                                                           \
    nrhs = NAME##_nrhs;                                                                                                                              \
  }

//
// THE PYBIND11 INTERFACE IMPLEMENTATION
//
auto factor(py::array_t<double, py::array::c_style | py::array::forcecast> U, py::array_t<double, py::array::c_style | py::array::forcecast> P,
            py::array_t<double, py::array::c_style | py::array::forcecast> d, py::array_t<double, py::array::c_style | py::array::forcecast> W) {
  SETUP_BASE_MATRICES;
  Eigen::Index flag = 0;
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    GET_ROW_MAJOR(SIZE);                                                                                                                             \
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

auto solve(py::array_t<double, py::array::c_style | py::array::forcecast> U, py::array_t<double, py::array::c_style | py::array::forcecast> P,
           py::array_t<double, py::array::c_style | py::array::forcecast> d, py::array_t<double, py::array::c_style | py::array::forcecast> W,
           py::array_t<double, py::array::c_style | py::array::forcecast> Z) {
  SETUP_BASE_MATRICES;
  py::size_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    GET_ROW_MAJOR(SIZE);                                                                                                                             \
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

auto norm(py::array_t<double, py::array::c_style | py::array::forcecast> U, py::array_t<double, py::array::c_style | py::array::forcecast> P,
          py::array_t<double, py::array::c_style | py::array::forcecast> d, py::array_t<double, py::array::c_style | py::array::forcecast> W,
          py::array_t<double, py::array::c_style | py::array::forcecast> Z) {
  SETUP_BASE_MATRICES;
  py::size_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
  if (nrhs != 1) throw std::runtime_error("Z must be a vector");
  Eigen::Matrix<double, 1, 1> norm_;
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    GET_ROW_MAJOR(SIZE);                                                                                                                             \
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

auto matmul(py::array_t<double, py::array::c_style | py::array::forcecast> d, py::array_t<double, py::array::c_style | py::array::forcecast> U,
            py::array_t<double, py::array::c_style | py::array::forcecast> W, py::array_t<double, py::array::c_style | py::array::forcecast> P,
            py::array_t<double, py::array::c_style | py::array::forcecast> Y, py::array_t<double, py::array::c_style | py::array::forcecast> Z) {
  SETUP_BASE_MATRICES;
  py::size_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    GET_ROW_MAJOR(SIZE);                                                                                                                             \
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

auto dot_tril(py::array_t<double, py::array::c_style | py::array::forcecast> U, py::array_t<double, py::array::c_style | py::array::forcecast> P,
              py::array_t<double, py::array::c_style | py::array::forcecast> d, py::array_t<double, py::array::c_style | py::array::forcecast> W,
              py::array_t<double, py::array::c_style | py::array::forcecast> Z) {
  SETUP_BASE_MATRICES;
  py::size_t nrhs = 0;
  SETUP_RHS_MATRIX(Z);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    GET_ROW_MAJOR(SIZE);                                                                                                                             \
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

#undef UNWRAP_CASES

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(driver, m) {
  m.doc() = R"doc(
    The computation engine for celerite2

    These functions are low level and you shouldn't generally need or want to call them as a user.
)doc";

  py::register_exception<celerite2::driver::linalg_exception>(m, "LinAlgError");

  m.def("factor", &celerite2::driver::factor, "Compute the Cholesky factor of a celerite system", py::arg("U"), py::arg("P"), py::arg("d"),
        py::arg("W"));
  m.def("solve", &celerite2::driver::solve, "Solve a celerite system using the output of `factor`", py::arg("U"), py::arg("P"), py::arg("d"),
        py::arg("W"), py::arg("Z"));
  m.def("norm", &celerite2::driver::norm, "Compute the norm of a celerite system applied to a vector", py::arg("U"), py::arg("P"), py::arg("d"),
        py::arg("W"), py::arg("z"));
  m.def("matmul", &celerite2::driver::matmul, "Dot a celerite system into a matrix or vector", py::arg("a"), py::arg("U"), py::arg("W"), py::arg("P"),
        py::arg("Y"), py::arg("Z"));
  m.def("dot_tril", &celerite2::driver::dot_tril, "Dot the Cholesky factor celerite system into a matrix or vector", py::arg("U"), py::arg("P"),
        py::arg("d"), py::arg("W"), py::arg("Z"));

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
