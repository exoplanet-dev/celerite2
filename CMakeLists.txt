cmake_minimum_required(VERSION 3.15...3.27)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

include_directories(
    "c++/include"
    "c++/vendor/eigen"
    "python/celerite2")

pybind11_add_module(driver "python/celerite2/driver.cpp")
target_compile_features(driver PUBLIC cxx_std_14)
install(TARGETS driver LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

pybind11_add_module(backprop "python/celerite2/backprop.cpp")
target_compile_features(backprop PUBLIC cxx_std_14)
install(TARGETS backprop LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

option(BUILD_JAX "Build JAX extension (requires jaxlib headers)" ON)
if(BUILD_JAX)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
        RESULT_VARIABLE JAXLIB_RES)
    if(JAXLIB_RES EQUAL 0 AND NOT "${XLA_DIR}" STREQUAL "")
        message(STATUS "Building JAX extension with XLA include: ${XLA_DIR}")
        pybind11_add_module(xla_ops "python/celerite2/jax/xla_ops.cpp")
        target_compile_features(xla_ops PUBLIC cxx_std_17)
        target_include_directories(xla_ops PUBLIC "${XLA_DIR}")
        install(TARGETS xla_ops LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/jax")
    else()
        message(STATUS "Skipping JAX extension (jax.ffi include_dir not found)")
    endif()
endif()
