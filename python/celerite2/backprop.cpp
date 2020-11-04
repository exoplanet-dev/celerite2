#include <pybind11/pybind11.h>
#include <cmath>
#include "driver.hpp"

namespace py = pybind11;

namespace celerite2 {
namespace driver {

auto factor_fwd(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> a,
                py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V, py::array_t<double, py::array::c_style> d,
                py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> S) {
  SETUP_BASE_MATRICES;

  GET_BUF_VEC(a, N);
  GET_BUF_MAT(V, N, J);
  GET_BUF_MAT(S, N, J * J);

  Eigen::Index flag = 0;

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(a_, abuf, N);
  VECTOR(d_, dbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, N, J);                                                                                                              \
    MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                                    \
    MATRIX((SIZE * SIZE), S_, Sbuf, N, (J * J));                                                                                                     \
    flag = celerite2::core::factor(t_, c_, a_, U_, V_, d_, W_, S_);                                                                                  \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
  if (flag) throw backprop_linalg_exception();
  return std::make_tuple(d, W, S);
}

auto factor_rev(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> a,
                py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> V, py::array_t<double, py::array::c_style> d,
                py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> S, py::array_t<double, py::array::c_style> bd,
                py::array_t<double, py::array::c_style> bW, py::array_t<double, py::array::c_style> bt, py::array_t<double, py::array::c_style> bc,
                py::array_t<double, py::array::c_style> ba, py::array_t<double, py::array::c_style> bU, py::array_t<double, py::array::c_style> bV) {
  SETUP_BASE_MATRICES;

  GET_BUF_VEC(a, N);
  GET_BUF_MAT(V, N, J);
  GET_BUF_MAT(S, N, J * J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  GET_BUF_VEC(bt, N);
  GET_BUF_VEC(bc, J);
  GET_BUF_VEC(ba, N);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_MAT(bV, N, J);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(a_, abuf, N);
  CONST_VECTOR(d_, dbuf, N);
  CONST_VECTOR(bd_, bdbuf, N);
  VECTOR(bt_, btbuf, N);
  VECTOR(ba_, babuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, V_, Vbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    CONST_MATRIX((SIZE * SIZE), S_, Sbuf, N, (J * J));                                                                                               \
    CONST_MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                            \
    COEFFS(SIZE, bc_, bcbuf, J);                                                                                                                     \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bV_, bVbuf, N, J);                                                                                                                  \
    celerite2::core::factor_rev(t_, c_, a_, U_, V_, d_, W_, S_, bd_, bW_, bt_, bc_, ba_, bU_, bV_);                                                  \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP

  return std::make_tuple(bt, bc, ba, bU, bV);
}

auto solve_fwd(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
               py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
               py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F,
               py::array_t<double, py::array::c_style> G) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(X);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);
  GET_BUF_MAT(G, N, nrhs * J);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      VECTOR(X_, Xbuf, N);                                                                                                                           \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                                  \
      MATRIX(SIZE, G_, Gbuf, N, J);                                                                                                                  \
      celerite2::core::solve(t_, c_, U_, d_, W_, Y_, X_, Z_, F_, G_);                                                                                \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      MATRIX(Eigen::Dynamic, X_, Xbuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, F_, Fbuf, N, (J * nrhs));                                                                                               \
      MATRIX(Eigen::Dynamic, G_, Gbuf, N, (J * nrhs));                                                                                               \
      celerite2::core::solve(t_, c_, U_, d_, W_, Y_, X_, Z_, F_, G_);                                                                                \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(X, Z, F, G);
}

auto solve_rev(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
               py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
               py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F,
               py::array_t<double, py::array::c_style> G, py::array_t<double, py::array::c_style> bX, py::array_t<double, py::array::c_style> bt,
               py::array_t<double, py::array::c_style> bc, py::array_t<double, py::array::c_style> bU, py::array_t<double, py::array::c_style> bd,
               py::array_t<double, py::array::c_style> bW, py::array_t<double, py::array::c_style> bY) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(X);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);
  GET_BUF_MAT(G, N, nrhs * J);

  SETUP_RHS_MATRIX(bX);

  GET_BUF_VEC(bt, N);
  GET_BUF_VEC(bc, J);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  SETUP_RHS_MATRIX(bY);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  VECTOR(bt_, btbuf, N);
  VECTOR(bd_, bdbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
                                                                                                                                                     \
    COEFFS(SIZE, bc_, bcbuf, J);                                                                                                                     \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                                  \
                                                                                                                                                     \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      CONST_VECTOR(X_, Xbuf, N);                                                                                                                     \
      CONST_VECTOR(Z_, Zbuf, N);                                                                                                                     \
      CONST_MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                            \
      CONST_MATRIX(SIZE, G_, Gbuf, N, J);                                                                                                            \
                                                                                                                                                     \
      CONST_VECTOR(bX_, bXbuf, N);                                                                                                                   \
      VECTOR(bY_, bYbuf, N);                                                                                                                         \
                                                                                                                                                     \
      celerite2::core::solve_rev(t_, c_, U_, d_, W_, Y_, X_, Z_, F_, G_, bX_, bt_, bc_, bU_, bd_, bW_, bY_);                                         \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, X_, Xbuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, F_, Fbuf, N, (J * nrhs));                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, G_, Gbuf, N, (J * nrhs));                                                                                         \
                                                                                                                                                     \
      CONST_MATRIX(Eigen::Dynamic, bX_, bXbuf, N, nrhs);                                                                                             \
      MATRIX(Eigen::Dynamic, bY_, bYbuf, N, nrhs);                                                                                                   \
                                                                                                                                                     \
      celerite2::core::solve_rev(t_, c_, U_, d_, W_, Y_, X_, Z_, F_, G_, bX_, bt_, bc_, bU_, bd_, bW_, bY_);                                         \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(bt, bc, bU, bd, bW, bY);
}

auto dot_tril_fwd(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
                  py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
                  py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                                  \
      celerite2::core::dot_tril(t_, c_, U_, d_, W_, Y_, Z_, F_);                                                                                     \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, F_, Fbuf, N, (J * nrhs));                                                                                               \
      celerite2::core::dot_tril(t_, c_, U_, d_, W_, Y_, Z_, F_);                                                                                     \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(Z, F);
}

auto dot_tril_rev(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
                  py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
                  py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F, py::array_t<double, py::array::c_style> bZ,
                  py::array_t<double, py::array::c_style> bt, py::array_t<double, py::array::c_style> bc, py::array_t<double, py::array::c_style> bU,
                  py::array_t<double, py::array::c_style> bd, py::array_t<double, py::array::c_style> bW,
                  py::array_t<double, py::array::c_style> bY) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);

  SETUP_RHS_MATRIX(bZ);

  GET_BUF_VEC(bt, N);
  GET_BUF_VEC(bc, J);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  SETUP_RHS_MATRIX(bY);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  VECTOR(bt_, btbuf, N);
  VECTOR(bd_, bdbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
                                                                                                                                                     \
    COEFFS(SIZE, bc_, bcbuf, J);                                                                                                                     \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                                  \
                                                                                                                                                     \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      CONST_VECTOR(Z_, Zbuf, N);                                                                                                                     \
      CONST_MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                            \
                                                                                                                                                     \
      CONST_VECTOR(bZ_, bZbuf, N);                                                                                                                   \
      VECTOR(bY_, bYbuf, N);                                                                                                                         \
                                                                                                                                                     \
      celerite2::core::dot_tril_rev(t_, c_, U_, d_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bd_, bW_, bY_);                                              \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, F_, Fbuf, N, J *nrhs);                                                                                            \
                                                                                                                                                     \
      CONST_MATRIX(Eigen::Dynamic, bZ_, bZbuf, N, nrhs);                                                                                             \
      MATRIX(Eigen::Dynamic, bY_, bYbuf, N, nrhs);                                                                                                   \
                                                                                                                                                     \
      celerite2::core::dot_tril_rev(t_, c_, U_, d_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bd_, bW_, bY_);                                              \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(bt, bc, bU, bd, bW, bY);
}

auto norm_fwd(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
              py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
              py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  if (nrhs != 1) throw std::invalid_argument("Y must be a vector");
  GET_BUF(X, 1);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  CONST_VECTOR(Y_, Ybuf, N);
  Eigen::Map<Eigen::Matrix<double, 1, 1>> X_((double *)Xbuf.ptr, 1, 1);
  VECTOR(Z_, Zbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                                    \
    celerite2::core::norm(t_, c_, U_, d_, W_, Y_, X_, Z_, F_);                                                                                       \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
  return std::make_tuple(X, Z, F);
}

auto norm_rev(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> U,
              py::array_t<double, py::array::c_style> d, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
              py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F,
              py::array_t<double, py::array::c_style> bX, py::array_t<double, py::array::c_style> bt, py::array_t<double, py::array::c_style> bc,
              py::array_t<double, py::array::c_style> bU, py::array_t<double, py::array::c_style> bd, py::array_t<double, py::array::c_style> bW,
              py::array_t<double, py::array::c_style> bY) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  if (nrhs != 1) throw std::invalid_argument("Y must be a vector");
  GET_BUF(X, 1);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);
  GET_BUF(bX, 1);

  GET_BUF_VEC(bt, N);
  GET_BUF_VEC(bc, J);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  SETUP_RHS_MATRIX(bY);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  CONST_VECTOR(Y_, Ybuf, N);
  Eigen::Map<const Eigen::Matrix<double, 1, 1>> X_((double *)Xbuf.ptr, 1, 1);
  CONST_VECTOR(Z_, Zbuf, N);
  VECTOR(bt_, btbuf, N);
  VECTOR(bd_, bdbuf, N);
  VECTOR(bY_, bYbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                              \
                                                                                                                                                     \
    Eigen::Map<const Eigen::Matrix<double, 1, 1>> bX_((double *)bXbuf.ptr, 1, 1);                                                                    \
    COEFFS(SIZE, bc_, bcbuf, J);                                                                                                                     \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                                  \
                                                                                                                                                     \
    celerite2::core::norm_rev(t_, c_, U_, d_, W_, Y_, X_, Z_, F_, bX_, bt_, bc_, bU_, bd_, bW_, bY_);                                                \
  }
  UNWRAP_CASES;
#undef FIXED_SIZE_MAP
  return std::make_tuple(bt, bc, bU, bd, bW, bY);
}

auto matmul_fwd(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> d,
                py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
                py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F,
                py::array_t<double, py::array::c_style> G) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(X);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);
  GET_BUF_MAT(G, N, nrhs * J);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      VECTOR(X_, Xbuf, N);                                                                                                                           \
      VECTOR(Z_, Zbuf, N);                                                                                                                           \
      MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                                  \
      MATRIX(SIZE, G_, Gbuf, N, J);                                                                                                                  \
      celerite2::core::matmul(t_, c_, d_, U_, W_, Y_, X_, Z_, F_, G_);                                                                               \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      MATRIX(Eigen::Dynamic, X_, Xbuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                                     \
      MATRIX(Eigen::Dynamic, F_, Fbuf, N, (J * nrhs));                                                                                               \
      MATRIX(Eigen::Dynamic, G_, Gbuf, N, (J * nrhs));                                                                                               \
      celerite2::core::matmul(t_, c_, d_, U_, W_, Y_, X_, Z_, F_, G_);                                                                               \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(X, Z, F, G);
}

auto matmul_rev(py::array_t<double, py::array::c_style> t, py::array_t<double, py::array::c_style> c, py::array_t<double, py::array::c_style> d,
                py::array_t<double, py::array::c_style> U, py::array_t<double, py::array::c_style> W, py::array_t<double, py::array::c_style> Y,
                py::array_t<double, py::array::c_style> X, py::array_t<double, py::array::c_style> Z, py::array_t<double, py::array::c_style> F,
                py::array_t<double, py::array::c_style> G, py::array_t<double, py::array::c_style> bX, py::array_t<double, py::array::c_style> bt,
                py::array_t<double, py::array::c_style> bc, py::array_t<double, py::array::c_style> bd, py::array_t<double, py::array::c_style> bU,
                py::array_t<double, py::array::c_style> bW, py::array_t<double, py::array::c_style> bY) {
  SETUP_BASE_MATRICES;

  ssize_t nrhs = 0;
  SETUP_RHS_MATRIX(Y);
  SETUP_RHS_MATRIX(X);
  SETUP_RHS_MATRIX(Z);
  GET_BUF_MAT(F, N, nrhs * J);
  GET_BUF_MAT(G, N, nrhs * J);

  SETUP_RHS_MATRIX(bX);

  GET_BUF_VEC(bt, N);
  GET_BUF_VEC(bc, J);
  GET_BUF_MAT(bU, N, J);
  GET_BUF_VEC(bd, N);
  GET_BUF_MAT(bW, N, J);
  SETUP_RHS_MATRIX(bY);

  CONST_VECTOR(t_, tbuf, N);
  CONST_VECTOR(d_, dbuf, N);
  VECTOR(bt_, btbuf, N);
  VECTOR(bd_, bdbuf, N);

#define FIXED_SIZE_MAP(SIZE)                                                                                                                         \
  {                                                                                                                                                  \
    CONST_COEFFS(SIZE, c_, cbuf, J);                                                                                                                 \
    CONST_MATRIX(SIZE, U_, Ubuf, N, J);                                                                                                              \
    CONST_MATRIX(SIZE, W_, Wbuf, N, J);                                                                                                              \
                                                                                                                                                     \
    COEFFS(SIZE, bc_, bcbuf, J);                                                                                                                     \
    MATRIX(SIZE, bU_, bUbuf, N, J);                                                                                                                  \
    MATRIX(SIZE, bW_, bWbuf, N, J);                                                                                                                  \
                                                                                                                                                     \
    if (nrhs == 1) {                                                                                                                                 \
      CONST_VECTOR(Y_, Ybuf, N);                                                                                                                     \
      CONST_VECTOR(X_, Xbuf, N);                                                                                                                     \
      CONST_VECTOR(Z_, Zbuf, N);                                                                                                                     \
      CONST_MATRIX(SIZE, F_, Fbuf, N, J);                                                                                                            \
      CONST_MATRIX(SIZE, G_, Gbuf, N, J);                                                                                                            \
                                                                                                                                                     \
      CONST_VECTOR(bX_, bXbuf, N);                                                                                                                   \
      VECTOR(bY_, bYbuf, N);                                                                                                                         \
                                                                                                                                                     \
      celerite2::core::matmul_rev(t_, c_, d_, U_, W_, Y_, X_, Z_, F_, G_, bX_, bt_, bc_, bd_, bU_, bW_, bY_);                                        \
    } else {                                                                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, Y_, Ybuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, X_, Xbuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, Z_, Zbuf, N, nrhs);                                                                                               \
      CONST_MATRIX(Eigen::Dynamic, F_, Fbuf, N, (J * nrhs));                                                                                         \
      CONST_MATRIX(Eigen::Dynamic, G_, Gbuf, N, (J * nrhs));                                                                                         \
                                                                                                                                                     \
      CONST_MATRIX(Eigen::Dynamic, bX_, bXbuf, N, nrhs);                                                                                             \
      MATRIX(Eigen::Dynamic, bY_, bYbuf, N, nrhs);                                                                                                   \
                                                                                                                                                     \
      celerite2::core::matmul_rev(t_, c_, d_, U_, W_, Y_, X_, Z_, F_, G_, bX_, bt_, bc_, bd_, bU_, bW_, bY_);                                        \
    }                                                                                                                                                \
  }
  UNWRAP_CASES_FEW;
#undef FIXED_SIZE_MAP
  return std::make_tuple(bt, bc, bd, bU, bW, bY);
}

} // namespace driver
} // namespace celerite2

PYBIND11_MODULE(backprop, m) {

  py::register_exception<celerite2::driver::backprop_linalg_exception>(m, "LinAlgError");

  m.def("factor_fwd", &celerite2::driver::factor_fwd, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("a").noconvert(),
        py::arg("U").noconvert(), py::arg("V").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("S").noconvert());
  m.def("factor_rev", &celerite2::driver::factor_rev, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("a").noconvert(),
        py::arg("U").noconvert(), py::arg("V").noconvert(), py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("S").noconvert(),
        py::arg("bd").noconvert(), py::arg("bW").noconvert(), py::arg("bt").noconvert(), py::arg("bc").noconvert(), py::arg("ba").noconvert(),
        py::arg("b").noconvert(), py::arg("V").noconvert());

  m.def("solve_fwd", &celerite2::driver::solve_fwd, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert(), py::arg("G").noconvert());
  m.def("solve_rev", &celerite2::driver::solve_rev, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert(), py::arg("G").noconvert(), py::arg("bX").noconvert(), py::arg("bt").noconvert(), py::arg("bc").noconvert(),
        py::arg("bU").noconvert(), py::arg("bd").noconvert(), py::arg("bW").noconvert(), py::arg("bY").noconvert());

  m.def("norm_fwd", &celerite2::driver::norm_fwd, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert());
  m.def("norm_rev", &celerite2::driver::norm_rev, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert(), py::arg("bX").noconvert(), py::arg("bt").noconvert(), py::arg("bc").noconvert(), py::arg("bU").noconvert(),
        py::arg("bd").noconvert(), py::arg("bW").noconvert(), py::arg("bY").noconvert());

  m.def("dot_tril_fwd", &celerite2::driver::dot_tril_fwd, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("Z").noconvert(), py::arg("F").noconvert());
  m.def("dot_tril_rev", &celerite2::driver::dot_tril_rev, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("U").noconvert(),
        py::arg("d").noconvert(), py::arg("W").noconvert(), py::arg("Y").noconvert(), py::arg("Z").noconvert(), py::arg("F").noconvert(),
        py::arg("bZ").noconvert(), py::arg("bt").noconvert(), py::arg("bc").noconvert(), py::arg("bU").noconvert(), py::arg("bd").noconvert(),
        py::arg("bW").noconvert(), py::arg("bY").noconvert());

  m.def("matmul_fwd", &celerite2::driver::matmul_fwd, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("a").noconvert(),
        py::arg("U").noconvert(), py::arg("V").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert(), py::arg("G").noconvert());
  m.def("matmul_rev", &celerite2::driver::matmul_rev, py::arg("t").noconvert(), py::arg("c").noconvert(), py::arg("a").noconvert(),
        py::arg("U").noconvert(), py::arg("V").noconvert(), py::arg("Y").noconvert(), py::arg("X").noconvert(), py::arg("Z").noconvert(),
        py::arg("F").noconvert(), py::arg("G").noconvert(), py::arg("bX").noconvert(), py::arg("bt").noconvert(), py::arg("bc").noconvert(),
        py::arg("ba").noconvert(), py::arg("bU").noconvert(), py::arg("bV").noconvert(), py::arg("bY").noconvert());

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
