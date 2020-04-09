#ifndef _CELERITE2_UTILS_HPP_DEFINED_
#define _CELERITE2_UTILS_HPP_DEFINED_

#include <Eigen/Core>
#include <cassert>

namespace celerite2 {
namespace utils {

#define ASSERT_SAME_OR_DYNAMIC(VAL, J) assert((VAL == Eigen::Dynamic) || (VAL == J))

// adapted from https://academy.realm.io/posts/how-we-beat-cpp-stl-binary-search/
template <typename T>
inline int search_sorted(const Eigen::MatrixBase<T> &x, const typename T::Scalar &value) {
  const int N = x.rows();
  int low = -1, high = N;
  while (high - low > 1) {
    int probe = (low + high) / 2;
    auto v    = x(probe);
    if (v > value)
      high = probe;
    else
      low = probe;
  }
  return high;
}

} // namespace utils
} // namespace celerite2

#endif // _CELERITE2_UTILS_HPP_DEFINED_
