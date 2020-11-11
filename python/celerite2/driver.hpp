#ifndef _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_
#define _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <exception>

#include <celerite2/celerite2.h>

namespace celerite2 {
namespace driver {

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

};     // namespace driver
};     // namespace celerite2
#endif // _CELERITE2_PYTHON_DRIVER_HPP_DEFINED_
