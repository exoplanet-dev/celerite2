# Make sure that we can find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen version: ${Eigen3_VERSION}")
message(STATUS "Eigen version: ${EIGEN3_INCLUDE_DIR}")

include_directories(${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.cpp)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  target_compile_options(${test_name} PUBLIC -fPIC -std=c++14)
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  target_link_libraries(${test_name} Eigen3::Eigen)
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
endforeach()
