#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <celerite2/celerite2.h>
#include "../driver.hpp"

#define GET_SIZES                                                                                                                                    \
  int base_index = 2;                                                                                                                                \
  const int N    = *reinterpret_cast<const int *>(in[0]);                                                                                            \
  const int J    = *reinterpret_cast<const int *>(in[1]);                                                                                            \
  void **out     = reinterpret_cast<void **>(out_tuple);

#define GET_NRHS                                                                                                                                     \
  const int nrhs = *reinterpret_cast<const int *>(in[2]);                                                                                            \
  base_index     = 3;

#undef COEFFS
#undef VECTOR
#undef MATRIX
#undef CONST_COEFFS
#undef CONST_VECTOR
#undef CONST_MATRIX

#define COEFFS(SIZE, NAME, INDEX, ROWS)                                                                                                              \
  double *NAME##_base = reinterpret_cast<double *>(out[base_index + INDEX]);                                                                         \
  Eigen::Map<Eigen::Matrix<double, SIZE, 1>> NAME(NAME##_base, ROWS, 1);
#define VECTOR(NAME, INDEX, ROWS)                                                                                                                    \
  double *NAME##_base = reinterpret_cast<double *>(out[base_index + INDEX]);                                                                         \
  Eigen::Map<Eigen::VectorXd> NAME(NAME##_base, ROWS, 1);
#define MATRIX(SIZE, NAME, INDEX, ROWS, COLS)                                                                                                        \
  double *NAME##_base = reinterpret_cast<double *>(out[base_index + INDEX]);                                                                         \
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, celerite2::driver::order<SIZE>::value>> NAME(NAME##_base, ROWS, COLS);

#define CONST_COEFFS(SIZE, NAME, INDEX, ROWS)                                                                                                        \
  const double *NAME##_base = reinterpret_cast<const double *>(in[INDEX]);                                                                           \
  Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> NAME(NAME##_base, ROWS, 1);
#define CONST_VECTOR(NAME, INDEX, ROWS)                                                                                                              \
  const double *NAME##_base = reinterpret_cast<const double *>(in[INDEX]);                                                                           \
  Eigen::Map<const Eigen::VectorXd> NAME(NAME##_base, ROWS, 1);
#define CONST_MATRIX(SIZE, NAME, INDEX, ROWS, COLS)                                                                                                  \
  const double *NAME##_base = reinterpret_cast<const double *>(in[INDEX]);                                                                           \
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, celerite2::driver::order<SIZE>::value>> NAME(NAME##_base, ROWS, COLS);

namespace py = pybind11;

const void factor(void *out_tuple, const void **in) {
  GET_SIZES
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_VECTOR(a, 2, N);                                                                                                                           \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
                                                                                                                                                     \
    VECTOR(d, 0, N);                                                                                                                                 \
    MATRIX(SIZE, W, 1, N, J);                                                                                                                        \
    MATRIX((SIZE * SIZE), S, 2, N, J *J);                                                                                                            \
    Eigen::Index flag = celerite2::core::factor(t, c, a, U, V, d, W, S);                                                                             \
    if (flag) d.setZero();                                                                                                                           \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void factor_rev(void *out_tuple, const void **in) {
  GET_SIZES
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_VECTOR(a, 2, N);                                                                                                                           \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    CONST_VECTOR(d, 5, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 6, N, J);                                                                                                                  \
    CONST_MATRIX((SIZE * SIZE), S, 7, N, J *J);                                                                                                      \
    CONST_VECTOR(bd, 8, N);                                                                                                                          \
    CONST_MATRIX(SIZE, bW, 9, N, J);                                                                                                                 \
                                                                                                                                                     \
    VECTOR(bt, 0, N);                                                                                                                                \
    COEFFS(SIZE, bc, 1, J);                                                                                                                          \
    VECTOR(ba, 2, N);                                                                                                                                \
    MATRIX(SIZE, bU, 3, N, J);                                                                                                                       \
    MATRIX(SIZE, bV, 4, N, J);                                                                                                                       \
    celerite2::core::factor_rev(t, c, a, U, V, d, W, S, bd, bW, bt, bc, ba, bU, bV);                                                                 \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void solve(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      VECTOR(Z, 1, N);                                                                                                                               \
      MATRIX(SIZE, F, 2, N, J);                                                                                                                      \
      MATRIX(SIZE, G, 3, N, J);                                                                                                                      \
      celerite2::core::solve(t, c, U, d, W, Y, X, Z, F, G);                                                                                          \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, Z, 1, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 2, N, (J * nrhs));                                                                                                   \
      MATRIX(Eigen::Dynamic, G, 3, N, (J * nrhs));                                                                                                   \
      celerite2::core::solve(t, c, U, d, W, Y, X, Z, F, G);                                                                                          \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void solve_rev(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
                                                                                                                                                     \
    VECTOR(bt, 0, N);                                                                                                                                \
    COEFFS(SIZE, bc, 1, J);                                                                                                                          \
    VECTOR(bd, 2, N);                                                                                                                                \
    MATRIX(SIZE, bU, 3, N, J);                                                                                                                       \
    MATRIX(SIZE, bW, 4, N, J);                                                                                                                       \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      CONST_VECTOR(X, 6, N);                                                                                                                         \
      CONST_VECTOR(Z, 7, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 8, N, J);                                                                                                                \
      CONST_MATRIX(SIZE, G, 9, N, J);                                                                                                                \
      CONST_VECTOR(bX, 10, N);                                                                                                                       \
      VECTOR(bY, 5, N);                                                                                                                              \
      celerite2::core::solve_rev(t, c, U, d, W, Y, X, Z, F, G, bX, bt, bc, bU, bd, bW, bY);                                                          \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 6, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, Z, 7, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 8, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, G, 9, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, bX, 10, N, nrhs);                                                                                                 \
      MATRIX(Eigen::Dynamic, bY, 5, N, nrhs);                                                                                                        \
      celerite2::core::solve_rev(t, c, U, d, W, Y, X, Z, F, G, bX, bt, bc, bU, bd, bW, bY);                                                          \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void norm(void *out_tuple, const void **in) {
  GET_SIZES
  double *X_base = reinterpret_cast<double *>(out[0]);
  Eigen::Map<Eigen::Matrix<double, 1, 1>> X(X_base, 1, 1);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
    CONST_VECTOR(Y, 5, N);                                                                                                                           \
    VECTOR(Z, 1, N);                                                                                                                                 \
    MATRIX(SIZE, F, 2, N, J);                                                                                                                        \
    celerite2::core::norm(t, c, U, d, W, Y, X, Z, F);                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void norm_rev(void *out_tuple, const void **in) {
  GET_SIZES

  const double *X_base = reinterpret_cast<const double *>(in[base_index + 6]);
  Eigen::Map<const Eigen::Matrix<double, 1, 1>> X(X_base, 1, 1);
  const double *bX_base = reinterpret_cast<const double *>(in[10]);
  Eigen::Map<const Eigen::Matrix<double, 1, 1>> bX(bX_base, 1, 1);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
    CONST_VECTOR(Y, 5, N);                                                                                                                           \
    CONST_VECTOR(Z, 7, N);                                                                                                                           \
    CONST_MATRIX(SIZE, F, 8, N, J);                                                                                                                  \
    VECTOR(bt, 0, N);                                                                                                                                \
    COEFFS(SIZE, bc, 1, J);                                                                                                                          \
    MATRIX(SIZE, bU, 2, N, J);                                                                                                                       \
    VECTOR(bd, 3, N);                                                                                                                                \
    MATRIX(SIZE, bW, 4, N, J);                                                                                                                       \
    VECTOR(bY, 5, N);                                                                                                                                \
    celerite2::core::norm_rev(t, c, U, d, W, Y, X, Z, F, bX, bt, bc, bU, bd, bW, bY);                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void dot_tril(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      MATRIX(SIZE, F, 1, N, J);                                                                                                                      \
      celerite2::core::dot_tril(t, c, U, d, W, Y, X, F);                                                                                             \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 1, N, (J * nrhs));                                                                                                   \
      celerite2::core::dot_tril(t, c, U, d, W, Y, X, F);                                                                                             \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void dot_tril_rev(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_VECTOR(d, 3, N);                                                                                                                           \
    CONST_MATRIX(SIZE, W, 4, N, J);                                                                                                                  \
    VECTOR(bt, 0, N);                                                                                                                                \
    COEFFS(SIZE, bc, 1, J);                                                                                                                          \
    MATRIX(SIZE, bU, 2, N, J);                                                                                                                       \
    VECTOR(bd, 3, N);                                                                                                                                \
    MATRIX(SIZE, bW, 4, N, J);                                                                                                                       \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      CONST_VECTOR(X, 6, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 7, N, J);                                                                                                                \
      CONST_VECTOR(bX, 8, N);                                                                                                                        \
      VECTOR(bY, 5, N);                                                                                                                              \
      celerite2::core::dot_tril_rev(t, c, U, d, W, Y, X, F, bX, bt, bc, bU, bd, bW, bY);                                                             \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 6, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 7, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, bX, 8, N, nrhs);                                                                                                  \
      MATRIX(Eigen::Dynamic, bY, 5, N, nrhs);                                                                                                        \
      celerite2::core::dot_tril_rev(t, c, U, d, W, Y, X, F, bX, bt, bc, bU, bd, bW, bY);                                                             \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void matmul(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_VECTOR(a, 2, N);                                                                                                                           \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      VECTOR(Z, 1, N);                                                                                                                               \
      MATRIX(SIZE, F, 2, N, J);                                                                                                                      \
      MATRIX(SIZE, G, 3, N, J);                                                                                                                      \
      celerite2::core::matmul(t, c, a, U, V, Y, X, Z, F, G);                                                                                         \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, Z, 1, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 2, N, (J * nrhs));                                                                                                   \
      MATRIX(Eigen::Dynamic, G, 3, N, (J * nrhs));                                                                                                   \
      celerite2::core::matmul(t, c, a, U, V, Y, X, Z, F, G);                                                                                         \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void matmul_rev(void *out_tuple, const void **in) {
  GET_SIZES
  GET_NRHS

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_VECTOR(t, 0, N);                                                                                                                           \
    CONST_COEFFS(SIZE, c, 1, J);                                                                                                                     \
    CONST_VECTOR(a, 2, N);                                                                                                                           \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    VECTOR(bt, 0, N);                                                                                                                                \
    COEFFS(SIZE, bc, 1, J);                                                                                                                          \
    VECTOR(ba, 2, N);                                                                                                                                \
    MATRIX(SIZE, bU, 3, N, J);                                                                                                                       \
    MATRIX(SIZE, bV, 4, N, J);                                                                                                                       \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 5, N);                                                                                                                         \
      CONST_VECTOR(X, 6, N);                                                                                                                         \
      CONST_VECTOR(Z, 7, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 8, N, J);                                                                                                                \
      CONST_MATRIX(SIZE, G, 9, N, J);                                                                                                                \
      CONST_VECTOR(bX, 10, N);                                                                                                                       \
      VECTOR(bY, 5, N);                                                                                                                              \
      celerite2::core::matmul_rev(t, c, a, U, V, Y, X, Z, F, G, bX, bt, bc, ba, bU, bV, bY);                                                         \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 5, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 6, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, Z, 7, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 8, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, G, 9, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, bX, 10, N, nrhs);                                                                                                 \
      MATRIX(Eigen::Dynamic, bY, 5, N, nrhs);                                                                                                        \
      celerite2::core::matmul_rev(t, c, a, U, V, Y, X, Z, F, G, bX, bt, bc, ba, bU, bV, bY);                                                         \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

// const void conditional_mean(void *out_tuple, const void **in) {
//   GET_SIZES
//   const int M = *reinterpret_cast<const int *>(in[2]);

//   CONST_VECTOR(Z, 6, N);
//   const std::int64_t *inds_base = reinterpret_cast<const std::int64_t *>(in[9]);
//   Eigen::Map<const Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>> inds(inds_base, M, 1);

//   VECTOR(mu, 0, M);

// #define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
//   {                                                                                                                                                  \
//     CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
//     CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
//     CONST_MATRIX(SIZE, P, 5, N - 1, J);                                                                                                              \
//     CONST_MATRIX(SIZE, U_star, 7, M, J);                                                                                                             \
//     CONST_MATRIX(SIZE, V_star, 8, M, J);                                                                                                             \
//     celerite2::core::conditional_mean(U, V, P, Z, U_star, V_star, inds, mu);                                                                         \
//   }
//   UNWRAP_CASES;
// #undef FIXED_SIZE_MAP
// }

PYBIND11_MODULE(xla_ops, m) {
  m.def("factor", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&factor, name);
  });
  m.def("factor_rev", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&factor_rev, name);
  });
  m.def("solve", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&solve, name);
  });
  m.def("solve_rev", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&solve_rev, name);
  });
  m.def("norm", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&norm, name);
  });
  m.def("norm_rev", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&norm_rev, name);
  });
  m.def("dot_tril", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&dot_tril, name);
  });
  m.def("dot_tril_rev", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&dot_tril_rev, name);
  });
  m.def("matmul", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&matmul, name);
  });
  m.def("matmul_rev", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&matmul_rev, name);
  });
  // m.def("conditional_mean", []() {
  //   const char *name = "xla._CUSTOM_CALL_TARGET";
  //   return py::capsule((void *)&conditional_mean, name);
  // });
}
