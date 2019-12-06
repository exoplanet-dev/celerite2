include_directories(${CMAKE_CURRENT_LIST_DIR})

file(GLOB_RECURSE all_tests RELATIVE ${CMAKE_CURRENT_LIST_DIR} *.cpp)

foreach(test ${all_tests})
  get_filename_component(test_name ${test} NAME_WE)
  get_filename_component(test_dir ${test} DIRECTORY)
  add_executable(${test_name} ${test})
  target_compile_options(${test_name} PUBLIC -fPIC -std=c++14)
  set_property(TARGET ${test_name} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
  add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${test_dir})
endforeach()
