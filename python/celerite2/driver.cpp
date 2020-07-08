#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>
#include <cmath>

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

// This gets the buffer info for the standard celerite matrix inputs and checks the dimensions
#define SETUP_BASE_MATRICES                                                                                                                          \
  py::buffer_info Ubuf = U.request(), Pbuf = P.request(), dbuf = d.request(), Wbuf = W.request();                                                    \
  if (Ubuf.ndim != 2 || Pbuf.ndim != 2 || dbuf.ndim != 1 || Wbuf.ndim != 2) throw std::runtime_error("Invalid dimensions");                          \
  ssize_t N = Ubuf.shape[0], J = Ubuf.shape[1];                                                                                                      \
  if (N == 0 || J == 0) throw std::runtime_error("Dimensions can't be zero");                                                                        \
  if (Pbuf.shape[0] != N - 1 || Pbuf.shape[1] != J) throw std::runtime_error("Invalid shape: P");                                                    \
  if (dbuf.shape[0] != N) throw std::runtime_error("Invalid shape: d");                                                                              \
  if (Wbuf.shape[0] != N || Wbuf.shape[1] != J) throw std::runtime_error("Invalid shape: W");

// This gets the buffer info for a right hand side input and checks the dimensions
#define SETUP_RHS_MATRIX(NAME)                                                                                                                       \
  py::buffer_info NAME##buf = NAME.request();                                                                                                        \
  ssize_t NAME##_nrhs       = 1;                                                                                                                     \
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

#undef UNWRAP_CASES

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
