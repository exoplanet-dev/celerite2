#ifndef _CELERITE_UTILS_HPP_DEFINED_
#define _CELERITE_UTILS_HPP_DEFINED_

#include <Eigen/Core>
#include <cassert>

namespace celerite {
namespace utils {

#define ASSERT_SAME_OR_DYNAMIC(VAL, J) assert((VAL == Eigen::Dynamic) || (VAL == J))

constexpr const int get_compile_time_size(const int size1, const int size2) {
  return (size1 < size2) ? size2 : size1;
}

constexpr const int get_compile_time_size(const int size1, const int size2, const int size3) {
  return get_compile_time_size(get_compile_time_size(size1, size2), size3);
}

constexpr const int get_compile_time_size(const int size1, const int size2, const int size3,
                                          const int size4) {
  return get_compile_time_size(get_compile_time_size(size1, size2, size3), size4);
}

constexpr const int get_compile_time_size(const int size1, const int size2, const int size3,
                                          const int size4, const int size5) {
  return get_compile_time_size(get_compile_time_size(size1, size2, size3, size4), size5);
}

}  // namespace utils
}  // namespace celerite

#endif  // _CELERITE_UTILS_HPP_DEFINED_
