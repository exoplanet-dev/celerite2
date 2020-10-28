#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <celerite2/celerite2.h>
#include "../driver.hpp"

#include <iostream>

#define GET_SIZES                                                                                                                                    \
  const int N = *reinterpret_cast<const int *>(in[0]);                                                                                               \
  const int J = *reinterpret_cast<const int *>(in[1]);                                                                                               \
  void **out  = reinterpret_cast<void **>(out_tuple);

#define VECTOR(NAME, INDEX, ROWS)                                                                                                                    \
  double *NAME##_base = reinterpret_cast<double *>(out[INDEX]);                                                                                      \
  Eigen::Map<Eigen::VectorXd> NAME(NAME##_base, ROWS, 1);
#define MATRIX(SIZE, NAME, INDEX, ROWS, COLS)                                                                                                        \
  double *NAME##_base = reinterpret_cast<double *>(out[INDEX]);                                                                                      \
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, celerite2::driver::order<SIZE>::value>> NAME(NAME##_base, ROWS, COLS);

#define CONST_VECTOR(NAME, INDEX, ROWS)                                                                                                              \
  const double *NAME##_base = reinterpret_cast<const double *>(in[INDEX]);                                                                           \
  Eigen::Map<const Eigen::VectorXd> NAME(NAME##_base, ROWS, 1);
#define CONST_MATRIX(SIZE, NAME, INDEX, ROWS, COLS)                                                                                                  \
  const double *NAME##_base = reinterpret_cast<const double *>(in[INDEX]);                                                                           \
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, celerite2::driver::order<SIZE>::value>> NAME(NAME##_base, ROWS, COLS);

namespace py = pybind11;

const void factor(void *out_tuple, const void **in) {
  void **out  = reinterpret_cast<void **>(out_tuple);
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int J = *reinterpret_cast<const int *>(in[1]);

  CONST_VECTOR(a, 2, N);
  VECTOR(d, 0, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 5, N - 1, J);                                                                                                              \
    MATRIX(SIZE, W, 1, N, J);                                                                                                                        \
    MATRIX((SIZE * SIZE), S, 2, N, J *J);                                                                                                            \
    Eigen::Index flag = celerite2::core::factor(a, U, V, P, d, W, S);                                                                                \
    if (flag) d.setZero();                                                                                                                           \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void factor_rev(void *out_tuple, const void **in) {
  void **out  = reinterpret_cast<void **>(out_tuple);
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int J = *reinterpret_cast<const int *>(in[1]);

  CONST_VECTOR(a, 2, N);
  CONST_VECTOR(d, 6, N);
  CONST_VECTOR(bd, 9, N);
  VECTOR(ba, 0, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 5, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 7, N, J);                                                                                                                  \
    CONST_MATRIX((SIZE * SIZE), S, 8, N, J *J);                                                                                                      \
    CONST_MATRIX(SIZE, bW, 10, N, J);                                                                                                                \
    MATRIX(SIZE, bU, 1, N, J);                                                                                                                       \
    MATRIX(SIZE, bV, 2, N, J);                                                                                                                       \
    MATRIX(SIZE, bP, 3, N - 1, J);                                                                                                                   \
    celerite2::core::factor_rev(a, U, V, P, d, W, S, bd, bW, ba, bU, bV, bP);                                                                        \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void solve(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(d, 5, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 4, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 6, N, J);                                                                                                                  \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      VECTOR(Z, 1, N);                                                                                                                               \
      MATRIX(SIZE, F, 2, N, J);                                                                                                                      \
      MATRIX(SIZE, G, 3, N, J);                                                                                                                      \
      celerite2::core::solve(U, P, d, W, Y, X, Z, F, G);                                                                                             \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, Z, 1, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 2, N, (J * nrhs));                                                                                                   \
      MATRIX(Eigen::Dynamic, G, 3, N, (J * nrhs));                                                                                                   \
      celerite2::core::solve(U, P, d, W, Y, X, Z, F, G);                                                                                             \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void solve_rev(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(d, 5, N);
  VECTOR(bd, 2, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 4, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 6, N, J);                                                                                                                  \
    MATRIX(SIZE, bU, 0, N, J);                                                                                                                       \
    MATRIX(SIZE, bP, 1, N - 1, J);                                                                                                                   \
    MATRIX(SIZE, bW, 3, N, J);                                                                                                                       \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      CONST_VECTOR(X, 8, N);                                                                                                                         \
      CONST_VECTOR(Z, 9, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 10, N, J);                                                                                                               \
      CONST_MATRIX(SIZE, G, 11, N, J);                                                                                                               \
      CONST_VECTOR(bX, 12, N);                                                                                                                       \
      VECTOR(bY, 4, N);                                                                                                                              \
      celerite2::core::solve_rev(U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY);                                                                 \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 8, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, Z, 9, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 10, N, (J * nrhs));                                                                                            \
      CONST_MATRIX(Eigen::Dynamic, G, 11, N, (J * nrhs));                                                                                            \
      CONST_MATRIX(Eigen::Dynamic, bX, 12, N, nrhs);                                                                                                 \
      MATRIX(Eigen::Dynamic, bY, 4, N, nrhs);                                                                                                        \
      celerite2::core::solve_rev(U, P, d, W, Y, X, Z, F, G, bX, bU, bP, bd, bW, bY);                                                                 \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void norm(void *out_tuple, const void **in) {
  void **out  = reinterpret_cast<void **>(out_tuple);
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int J = *reinterpret_cast<const int *>(in[1]);

  CONST_VECTOR(d, 4, N);
  CONST_VECTOR(Y, 6, N);
  VECTOR(Z, 1, N);

  double *X_base = reinterpret_cast<double *>(out[0]);
  Eigen::Map<Eigen::Matrix<double, 1, 1>> X(X_base, 1, 1);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 3, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 5, N, J);                                                                                                                  \
    MATRIX(SIZE, F, 2, N, J);                                                                                                                        \
    celerite2::core::norm(U, P, d, W, Y, X, Z, F);                                                                                                   \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void norm_rev(void *out_tuple, const void **in) {
  void **out  = reinterpret_cast<void **>(out_tuple);
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int J = *reinterpret_cast<const int *>(in[1]);

  CONST_VECTOR(d, 4, N);
  CONST_VECTOR(Y, 6, N);
  CONST_VECTOR(Z, 8, N);
  VECTOR(bd, 2, N);
  VECTOR(bY, 4, N);

  const double *X_base = reinterpret_cast<const double *>(in[7]);
  Eigen::Map<const Eigen::Matrix<double, 1, 1>> X(X_base, 1, 1);
  const double *bX_base = reinterpret_cast<const double *>(in[10]);
  Eigen::Map<const Eigen::Matrix<double, 1, 1>> bX(bX_base, 1, 1);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 2, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 3, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 5, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, F, 9, N, J);                                                                                                                  \
    MATRIX(SIZE, bU, 0, N, J);                                                                                                                       \
    MATRIX(SIZE, bP, 1, N - 1, J);                                                                                                                   \
    MATRIX(SIZE, bW, 3, N, J);                                                                                                                       \
    celerite2::core::norm_rev(U, P, d, W, Y, X, Z, F, bX, bU, bP, bd, bW, bY);                                                                       \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void dot_tril(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(d, 5, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 4, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 6, N, J);                                                                                                                  \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      MATRIX(SIZE, F, 1, N, J);                                                                                                                      \
      celerite2::core::dot_tril(U, P, d, W, Y, X, F);                                                                                                \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 1, N, (J * nrhs));                                                                                                   \
      celerite2::core::dot_tril(U, P, d, W, Y, X, F);                                                                                                \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void dot_tril_rev(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(d, 5, N);
  VECTOR(bd, 2, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 4, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, W, 6, N, J);                                                                                                                  \
    MATRIX(SIZE, bU, 0, N, J);                                                                                                                       \
    MATRIX(SIZE, bP, 1, N - 1, J);                                                                                                                   \
    MATRIX(SIZE, bW, 3, N, J);                                                                                                                       \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      CONST_VECTOR(X, 8, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 9, N, J);                                                                                                                \
      CONST_VECTOR(bX, 10, N);                                                                                                                       \
      VECTOR(bY, 4, N);                                                                                                                              \
      celerite2::core::dot_tril_rev(U, P, d, W, Y, X, F, bX, bU, bP, bd, bW, bY);                                                                    \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 8, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 9, N, (J * nrhs));                                                                                             \
      CONST_MATRIX(Eigen::Dynamic, bX, 10, N, nrhs);                                                                                                 \
      MATRIX(Eigen::Dynamic, bY, 4, N, nrhs);                                                                                                        \
      celerite2::core::dot_tril_rev(U, P, d, W, Y, X, F, bX, bU, bP, bd, bW, bY);                                                                    \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void matmul(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(a, 3, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 4, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 5, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 6, N - 1, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      VECTOR(X, 0, N);                                                                                                                               \
      VECTOR(Z, 1, N);                                                                                                                               \
      MATRIX(SIZE, F, 2, N, J);                                                                                                                      \
      MATRIX(SIZE, G, 3, N, J);                                                                                                                      \
      celerite2::core::matmul(a, U, V, P, Y, X, Z, F, G);                                                                                            \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      MATRIX(Eigen::Dynamic, X, 0, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, Z, 1, N, nrhs);                                                                                                         \
      MATRIX(Eigen::Dynamic, F, 2, N, (J * nrhs));                                                                                                   \
      MATRIX(Eigen::Dynamic, G, 3, N, (J * nrhs));                                                                                                   \
      celerite2::core::matmul(a, U, V, P, Y, X, Z, F, G);                                                                                            \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void matmul_rev(void *out_tuple, const void **in) {
  void **out     = reinterpret_cast<void **>(out_tuple);
  const int N    = *reinterpret_cast<const int *>(in[0]);
  const int J    = *reinterpret_cast<const int *>(in[1]);
  const int nrhs = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(a, 3, N);
  VECTOR(ba, 0, N);
#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 4, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 5, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 6, N - 1, J);                                                                                                              \
    MATRIX(SIZE, bU, 1, N, J);                                                                                                                       \
    MATRIX(SIZE, bV, 2, N, J);                                                                                                                       \
    MATRIX(SIZE, bP, 3, N - 1, J);                                                                                                                   \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y, 7, N);                                                                                                                         \
      CONST_VECTOR(X, 8, N);                                                                                                                         \
      CONST_VECTOR(Z, 9, N);                                                                                                                         \
      CONST_MATRIX(SIZE, F, 10, N, J);                                                                                                               \
      CONST_MATRIX(SIZE, G, 11, N, J);                                                                                                               \
      CONST_VECTOR(bX, 12, N);                                                                                                                       \
      VECTOR(bY, 4, N);                                                                                                                              \
      celerite2::core::matmul_rev(a, U, V, P, Y, X, Z, F, G, bX, ba, bU, bV, bP, bY);                                                                \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y, 7, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, X, 8, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, Z, 9, N, nrhs);                                                                                                   \
      CONST_MATRIX(Eigen::Dynamic, F, 10, N, (J * nrhs));                                                                                            \
      CONST_MATRIX(Eigen::Dynamic, G, 11, N, (J * nrhs));                                                                                            \
      CONST_MATRIX(Eigen::Dynamic, bX, 12, N, nrhs);                                                                                                 \
      MATRIX(Eigen::Dynamic, bY, 4, N, nrhs);                                                                                                        \
      celerite2::core::matmul_rev(a, U, V, P, Y, X, Z, F, G, bX, ba, bU, bV, bP, bY);                                                                \
    }                                                                                                                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

const void conditional_mean(void *out_tuple, const void **in) {
  void **out  = reinterpret_cast<void **>(out_tuple);
  const int N = *reinterpret_cast<const int *>(in[0]);
  const int J = *reinterpret_cast<const int *>(in[1]);
  const int M = *reinterpret_cast<const int *>(in[2]);

  CONST_VECTOR(Z, 6, N);
  const std::int64_t *inds_base = reinterpret_cast<const std::int64_t *>(in[9]);
  Eigen::Map<const Eigen::Matrix<std::int64_t, Eigen::Dynamic, 1>> inds(inds_base, M, 1);

  VECTOR(mu, 0, M);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_MATRIX(SIZE, U, 3, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, V, 4, N, J);                                                                                                                  \
    CONST_MATRIX(SIZE, P, 5, N - 1, J);                                                                                                              \
    CONST_MATRIX(SIZE, U_star, 7, M, J);                                                                                                             \
    CONST_MATRIX(SIZE, V_star, 8, M, J);                                                                                                             \
    celerite2::core::conditional_mean(U, V, P, Z, U_star, V_star, inds, mu);                                                                         \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
}

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
  m.def("conditional_mean", []() {
    const char *name = "xla._CUSTOM_CALL_TARGET";
    return py::capsule((void *)&conditional_mean, name);
  });
}
