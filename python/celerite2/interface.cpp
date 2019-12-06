#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <celerite2/core.hpp>

namespace py = pybind11;

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

void factor(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
            py::array_t<double, py::array::c_style> W) {

  py::buffer_info Ubuf = U.request(), Pbuf = P.request(), dbuf = d.request(), Wbuf = W.request();

  if (Ubuf.ndim != 2 || Pbuf.ndim != 2 || dbuf.ndim != 1 || Wbuf.ndim != 2) throw std::runtime_error("Invalid dimensions");

  long N = Ubuf.shape[0], J = Ubuf.shape[1];
  if (N == 0 || J == 0) throw std::runtime_error("Dimensions can't be zero");
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::runtime_error("Invalid shape: P");
  if (dbuf.shape[0] != N) throw std::runtime_error("Invalid shape: d");
  if (Wbuf.shape[0] != N || Wbuf.shape[1] != J) throw std::runtime_error("Invalid shape: W");

  int flag = 0;

  // Insane hack to deal with fixed dimensions in small systems
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    constexpr int RowMajor = (SIZE == 1) ? Eigen::ColMajor : Eigen::RowMajor;                                                                        \
    typedef Eigen::Matrix<double, Eigen::Dynamic, SIZE, RowMajor> Matrix;                                                                            \
    typedef Eigen::Map<Matrix> MatrixMap;                                                                                                            \
    typedef Eigen::Map<const Matrix> ConstMatrixMap;                                                                                                 \
    typedef Eigen::Map<Eigen::VectorXd> VectorMap;                                                                                                   \
    ConstMatrixMap U_((double *)Ubuf.ptr, N, J);                                                                                                     \
    ConstMatrixMap P_((double *)Pbuf.ptr, N - 1, J);                                                                                                 \
    VectorMap d_((double *)dbuf.ptr, N, 1);                                                                                                          \
    MatrixMap W_((double *)Wbuf.ptr, N, J);                                                                                                          \
    flag = celerite::core::factor(U_, P_, d_, W_);                                                                                                   \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP

  if (flag) throw std::runtime_error("Linear algrbra error");
}

void solve(py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> P, py::array_t<double, py::array::c_style> d,
           py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Z) {

  py::buffer_info Ubuf = U.request(), Pbuf = P.request(), dbuf = d.request(), Wbuf = W.request(), Zbuf = Z.request();

  if (Ubuf.ndim != 2 || Pbuf.ndim != 2 || dbuf.ndim != 1 || Wbuf.ndim != 2) throw std::runtime_error("Invalid dimensions");

  long N = Ubuf.shape[0], J = Ubuf.shape[1];
  if (N == 0 || J == 0) throw std::runtime_error("Dimensions can't be zero");
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::runtime_error("Invalid shape: P");
  if (dbuf.shape[0] != N) throw std::runtime_error("Invalid shape: d");
  if (Wbuf.shape[0] != N || Wbuf.shape[1] != J) throw std::runtime_error("Invalid shape: W");

  long nrhs = 1;
  if (Zbuf.ndim == 2) {
    nrhs = Zbuf.shape[1];
  } else if (Zbuf.ndim != 1) {
    throw std::runtime_error("Z must be a matrix");
  }
  if (Zbuf.shape[0] != N) throw std::runtime_error("Invalid shape: Z");

    // Insane hack to deal with fixed dimensions in small systems
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    constexpr int RowMajor = (SIZE == 1) ? Eigen::ColMajor : Eigen::RowMajor;                                                                        \
    typedef Eigen::Matrix<double, Eigen::Dynamic, SIZE, RowMajor> Matrix;                                                                            \
    typedef Eigen::Map<Matrix> MatrixMap;                                                                                                            \
    typedef Eigen::Map<const Matrix> ConstMatrixMap;                                                                                                 \
    typedef Eigen::Map<const Eigen::VectorXd> ConstVectorMap;                                                                                        \
    ConstMatrixMap U_((double *)Ubuf.ptr, N, J);                                                                                                     \
    ConstMatrixMap P_((double *)Pbuf.ptr, N - 1, J);                                                                                                 \
    ConstVectorMap d_((double *)dbuf.ptr, N, 1);                                                                                                     \
    ConstMatrixMap W_((double *)Wbuf.ptr, N, J);                                                                                                     \
    MatrixMap Z_((double *)Zbuf.ptr, N, nrhs);                                                                                                       \
    celerite::core::solve(U_, P_, d_, W_, Z_);                                                                                                       \
  }
  UNWRAP_CASES
#undef FIXED_SIZE_MAP
}

#undef UNWRAP_CASES

PYBIND11_MODULE(interface, m) {
  m.doc() = R"doc(
  Celerite2

  )doc";

  m.def("factor", &factor, R"doc(

  )doc");

  m.def("solve", &solve, R"doc(

  )doc");

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
