cmake_minimum_required(VERSION 3.15...3.27)
project(
    ${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

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

pybind11_add_module(xla_ops "python/celerite2/jax/xla_ops.cpp")
target_compile_features(xla_ops PUBLIC cxx_std_14)
install(TARGETS xla_ops LIBRARY DESTINATION "${SKBUILD_PROJECT_NAME}/jax")
