cmake_minimum_required(VERSION 3.5)

# Don't build in-tree
if (CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! Create a build dir and remove ${CMAKE_SOURCE_DIR}/CMakeCache.txt")
endif()

project(Celerite2 VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Get the git hash & print status
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE PROJECT_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "${PROJECT_NAME} version : ${PROJECT_VERSION}")
message(STATUS "${PROJECT_NAME} Git hash: ${PROJECT_GIT_HASH}")

# Assert that Install directory is given and invalid.
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Type of build" FORCE)
endif()
message(STATUS "Build-type: ${CMAKE_BUILD_TYPE}")

# Export the list of compile-commands into compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # Create a 'linkable' target
# add_library(Celerite2 INTERFACE)

# # Depend on some obvious c++11 features so the dependency is transitively added dependents
# target_compile_features(Celerite2
#   INTERFACE
#     cxx_alignas
#     cxx_alignof
#     cxx_attributes
#     cxx_auto_type
#     cxx_constexpr
#     cxx_defaulted_functions
#     cxx_deleted_functions
#     cxx_final
#     cxx_lambdas
#     cxx_noexcept
#     cxx_override
#     cxx_range_for
#     cxx_rvalue_references
#     cxx_static_assert
#     cxx_strong_enums
#     cxx_trailing_return_types
#     cxx_unicode_literals
#     cxx_user_literals
#     cxx_variadic_macros
# )
# target_include_directories(Celerite2
#   INTERFACE
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
#     $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
# target_link_libraries(Celerite2 INTERFACE Eigen3::Eigen)

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

# Install the headers
install(
  DIRECTORY
    "include/"
  DESTINATION
    "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Tests
option(Build_Tests "Build tests" ON)
if(Build_Tests)
  enable_testing()
  add_subdirectory(test)
endif()
