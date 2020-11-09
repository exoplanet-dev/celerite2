// NOTE: This file was autogenerated
// NOTE: Changes should be made to the template

#include <pybind11/pybind11.h>
#include "../driver.hpp"

namespace py = pybind11;
using namespace celerite2::driver;


auto factor (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);

    const double *t = reinterpret_cast<const double *>(in[2]);
    const double *c = reinterpret_cast<const double *>(in[3]);
    const double *a = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    double *d = reinterpret_cast<double *>(out[0]);
    double *W = reinterpret_cast<double *>(out[1]);
    double *S = reinterpret_cast<double *>(out[2]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::VectorXd> a_(a, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    Eigen::Map<Eigen::VectorXd> d_(d, N, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> S_(S, N, J * J); \
    Eigen::Index flag = celerite2::core::factor(t_, c_, a_, U_, V_, d_, W_, S_); \
    if (flag) d_.setZero(); \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}
auto factor_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);

    const double *t = reinterpret_cast<const double *>(in[2]);
    const double *c = reinterpret_cast<const double *>(in[3]);
    const double *a = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    const double *d = reinterpret_cast<const double *>(in[7]);
    const double *W = reinterpret_cast<const double *>(in[8]);
    const double *S = reinterpret_cast<const double *>(in[9]);
    const double *bd = reinterpret_cast<const double *>(in[10]);
    const double *bW = reinterpret_cast<const double *>(in[11]);
    double *bt = reinterpret_cast<double *>(out[0]);
    double *bc = reinterpret_cast<double *>(out[1]);
    double *ba = reinterpret_cast<double *>(out[2]);
    double *bU = reinterpret_cast<double *>(out[3]);
    double *bV = reinterpret_cast<double *>(out[4]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::VectorXd> a_(a, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    Eigen::Map<const Eigen::VectorXd> d_(d, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, (SIZE * SIZE), order<(SIZE * SIZE)>::value>> S_(S, N, J * J); \
    Eigen::Map<const Eigen::VectorXd> bd_(bd, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bW_(bW, N, J); \
    Eigen::Map<Eigen::VectorXd> bt_(bt, N, 1); \
    Eigen::Map<Eigen::Matrix<double, SIZE, 1>> bc_(bc, J, 1); \
    Eigen::Map<Eigen::VectorXd> ba_(ba, N, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bU_(bU, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bV_(bV, N, J); \
    celerite2::core::factor_rev(t_, c_, a_, U_, V_, d_, W_, S_, bd_, bW_, bt_, bc_, ba_, bU_, bV_); \
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}


auto solve_lower (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *W = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Z_.setZero(); \
        celerite2::core::solve_lower(t_, c_, U_, W_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::solve_lower(t_, c_, U_, W_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}
auto solve_lower_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *W = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    const double *Z = reinterpret_cast<const double *>(in[8]);
    const double *F = reinterpret_cast<const double *>(in[9]);
    const double *bZ = reinterpret_cast<const double *>(in[10]);
    double *bt = reinterpret_cast<double *>(out[0]);
    double *bc = reinterpret_cast<double *>(out[1]);
    double *bU = reinterpret_cast<double *>(out[2]);
    double *bW = reinterpret_cast<double *>(out[3]);
    double *bY = reinterpret_cast<double *>(out[4]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    Eigen::Map<Eigen::VectorXd> bt_(bt, N, 1); \
    Eigen::Map<Eigen::Matrix<double, SIZE, 1>> bc_(bc, J, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bU_(bU, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bW_(bW, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<const Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Eigen::Map<const Eigen::VectorXd> bZ_(bZ, N, 1); \
        Eigen::Map<Eigen::VectorXd> bY_(bY, N, 1); \
        celerite2::core::solve_lower_rev(t_, c_, U_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bW_, bY_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bZ_(bZ, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bY_(bY, N, nrhs); \
        celerite2::core::solve_lower_rev(t_, c_, U_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bW_, bY_); \
    } \
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}


auto solve_upper (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *W = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Z_.setZero(); \
        celerite2::core::solve_upper(t_, c_, U_, W_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::solve_upper(t_, c_, U_, W_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}
auto solve_upper_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *W = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    const double *Z = reinterpret_cast<const double *>(in[8]);
    const double *F = reinterpret_cast<const double *>(in[9]);
    const double *bZ = reinterpret_cast<const double *>(in[10]);
    double *bt = reinterpret_cast<double *>(out[0]);
    double *bc = reinterpret_cast<double *>(out[1]);
    double *bU = reinterpret_cast<double *>(out[2]);
    double *bW = reinterpret_cast<double *>(out[3]);
    double *bY = reinterpret_cast<double *>(out[4]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> W_(W, N, J); \
    Eigen::Map<Eigen::VectorXd> bt_(bt, N, 1); \
    Eigen::Map<Eigen::Matrix<double, SIZE, 1>> bc_(bc, J, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bU_(bU, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bW_(bW, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<const Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Eigen::Map<const Eigen::VectorXd> bZ_(bZ, N, 1); \
        Eigen::Map<Eigen::VectorXd> bY_(bY, N, 1); \
        celerite2::core::solve_upper_rev(t_, c_, U_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bW_, bY_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bZ_(bZ, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bY_(bY, N, nrhs); \
        celerite2::core::solve_upper_rev(t_, c_, U_, W_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bW_, bY_); \
    } \
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}


auto matmul_lower (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Z_.setZero(); \
        celerite2::core::matmul_lower(t_, c_, U_, V_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::matmul_lower(t_, c_, U_, V_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}
auto matmul_lower_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    const double *Z = reinterpret_cast<const double *>(in[8]);
    const double *F = reinterpret_cast<const double *>(in[9]);
    const double *bZ = reinterpret_cast<const double *>(in[10]);
    double *bt = reinterpret_cast<double *>(out[0]);
    double *bc = reinterpret_cast<double *>(out[1]);
    double *bU = reinterpret_cast<double *>(out[2]);
    double *bV = reinterpret_cast<double *>(out[3]);
    double *bY = reinterpret_cast<double *>(out[4]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    Eigen::Map<Eigen::VectorXd> bt_(bt, N, 1); \
    Eigen::Map<Eigen::Matrix<double, SIZE, 1>> bc_(bc, J, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bU_(bU, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bV_(bV, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<const Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Eigen::Map<const Eigen::VectorXd> bZ_(bZ, N, 1); \
        Eigen::Map<Eigen::VectorXd> bY_(bY, N, 1); \
        celerite2::core::matmul_lower_rev(t_, c_, U_, V_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bV_, bY_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bZ_(bZ, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bY_(bY, N, nrhs); \
        celerite2::core::matmul_lower_rev(t_, c_, U_, V_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bV_, bY_); \
    } \
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}


auto matmul_upper (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Z_.setZero(); \
        celerite2::core::matmul_upper(t_, c_, U_, V_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::matmul_upper(t_, c_, U_, V_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}
auto matmul_upper_rev (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int J = *reinterpret_cast<const int *>(in[1]);
    const int nrhs = *reinterpret_cast<const int *>(in[2]);

    const double *t = reinterpret_cast<const double *>(in[3]);
    const double *c = reinterpret_cast<const double *>(in[4]);
    const double *U = reinterpret_cast<const double *>(in[5]);
    const double *V = reinterpret_cast<const double *>(in[6]);
    const double *Y = reinterpret_cast<const double *>(in[7]);
    const double *Z = reinterpret_cast<const double *>(in[8]);
    const double *F = reinterpret_cast<const double *>(in[9]);
    const double *bZ = reinterpret_cast<const double *>(in[10]);
    double *bt = reinterpret_cast<double *>(out[0]);
    double *bc = reinterpret_cast<double *>(out[1]);
    double *bU = reinterpret_cast<double *>(out[2]);
    double *bV = reinterpret_cast<double *>(out[3]);
    double *bY = reinterpret_cast<double *>(out[4]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t_(t, N, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, N, J); \
    Eigen::Map<Eigen::VectorXd> bt_(bt, N, 1); \
    Eigen::Map<Eigen::Matrix<double, SIZE, 1>> bc_(bc, J, 1); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bU_(bU, N, J); \
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> bV_(bV, N, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, N, 1); \
        Eigen::Map<const Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, N, J); \
        Eigen::Map<const Eigen::VectorXd> bZ_(bZ, N, 1); \
        Eigen::Map<Eigen::VectorXd> bY_(bY, N, 1); \
        celerite2::core::matmul_upper_rev(t_, c_, U_, V_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bV_, bY_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, N, J * nrhs); \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bZ_(bZ, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bY_(bY, N, nrhs); \
        celerite2::core::matmul_upper_rev(t_, c_, U_, V_, Y_, Z_, F_, bZ_, bt_, bc_, bU_, bV_, bY_); \
    } \
    }
    UNWRAP_CASES_FEW
#undef FIXED_SIZE_MAP
}


auto general_matmul_lower (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int M = *reinterpret_cast<const int *>(in[1]);
    const int J = *reinterpret_cast<const int *>(in[2]);
    const int nrhs = *reinterpret_cast<const int *>(in[3]);

    const double *t1 = reinterpret_cast<const double *>(in[4]);
    const double *t2 = reinterpret_cast<const double *>(in[5]);
    const double *c = reinterpret_cast<const double *>(in[6]);
    const double *U = reinterpret_cast<const double *>(in[7]);
    const double *V = reinterpret_cast<const double *>(in[8]);
    const double *Y = reinterpret_cast<const double *>(in[9]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t1_(t1, N, 1); \
    Eigen::Map<const Eigen::VectorXd> t2_(t2, M, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, M, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, M, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, M, J); \
        Z_.setZero(); \
        celerite2::core::general_matmul_lower(t1_, t2_, c_, U_, V_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, M, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, M, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::general_matmul_lower(t1_, t2_, c_, U_, V_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
#undef FIXED_SIZE_MAP
}

auto general_matmul_upper (void *out_tuple, const void **in) {
    void **out = reinterpret_cast<void **>(out_tuple);
    const int N = *reinterpret_cast<const int *>(in[0]);
    const int M = *reinterpret_cast<const int *>(in[1]);
    const int J = *reinterpret_cast<const int *>(in[2]);
    const int nrhs = *reinterpret_cast<const int *>(in[3]);

    const double *t1 = reinterpret_cast<const double *>(in[4]);
    const double *t2 = reinterpret_cast<const double *>(in[5]);
    const double *c = reinterpret_cast<const double *>(in[6]);
    const double *U = reinterpret_cast<const double *>(in[7]);
    const double *V = reinterpret_cast<const double *>(in[8]);
    const double *Y = reinterpret_cast<const double *>(in[9]);
    double *Z = reinterpret_cast<double *>(out[0]);
    double *F = reinterpret_cast<double *>(out[1]);

#define FIXED_SIZE_MAP(SIZE) \
    { \
    Eigen::Map<const Eigen::VectorXd> t1_(t1, N, 1); \
    Eigen::Map<const Eigen::VectorXd> t2_(t2, M, 1); \
    Eigen::Map<const Eigen::Matrix<double, SIZE, 1>> c_(c, J, 1); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> U_(U, N, J); \
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> V_(V, M, J); \
    if (nrhs == 1) { \
        Eigen::Map<const Eigen::VectorXd> Y_(Y, M, 1); \
        Eigen::Map<Eigen::VectorXd> Z_(Z, N, 1); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, SIZE, order<SIZE>::value>> F_(F, M, J); \
        Z_.setZero(); \
        celerite2::core::general_matmul_upper(t1_, t2_, c_, U_, V_, Y_, Z_, F_); \
    } else { \
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Y_(Y, M, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> Z_(Z, N, nrhs); \
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> F_(F, M, J * nrhs); \
        Z_.setZero(); \
        celerite2::core::general_matmul_upper(t1_, t2_, c_, U_, V_, Y_, Z_, F_); \
    } \
    }
    UNWRAP_CASES_MOST
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
    m.def("solve_lower", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_lower, name);
    });
    m.def("solve_lower_rev", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_lower_rev, name);
    });
    m.def("solve_upper", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_upper, name);
    });
    m.def("solve_upper_rev", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&solve_upper_rev, name);
    });
    m.def("matmul_lower", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&matmul_lower, name);
    });
    m.def("matmul_lower_rev", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&matmul_lower_rev, name);
    });
    m.def("matmul_upper", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&matmul_upper, name);
    });
    m.def("matmul_upper_rev", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&matmul_upper_rev, name);
    });
    m.def("general_matmul_lower", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&general_matmul_lower, name);
    });
    m.def("general_matmul_upper", []() {
        const char *name = "xla._CUSTOM_CALL_TARGET";
        return py::capsule((void *)&general_matmul_upper, name);
    });
}
